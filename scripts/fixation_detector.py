#!/usr/bin/env python

import rospy
import actionlib
import pandas as pd
import numpy as np
import ibmmpy.msg
import ibmmpy.ibmm_online
import ibmmpy.srv
import collections
import pickle
import rosbag
import os
import traceback

## Message generating/parsing
def gaze_data_point_from_msg(msg):
    data = {'x': msg.position.x, 'y': msg.position.y, 'confidence': msg.confidence, 'timestamp': msg.header.stamp.to_sec()}
    if not np.isnan(msg.position.z):
        data['z'] = msg.position.z
    return data

GAZE_DATA_COLUMNS = ['timestamp', 'confidence', 'x', 'y', 'z']

def gaze_data_from_msg(msg):
    return {
        'world': pd.DataFrame([gaze_data_point_from_msg(m) for m in msg.world_data], columns=GAZE_DATA_COLUMNS),
        'eyes': [pd.DataFrame([gaze_data_point_from_msg(m) for m in msg.eye0_data], columns=GAZE_DATA_COLUMNS),
                 pd.DataFrame([gaze_data_point_from_msg(m) for m in msg.eye1_data], columns=GAZE_DATA_COLUMNS)]
        }

def msg_from_gaze_data_point(data):
    msg = ibmmpy.msg.GazeDataPoint(confidence=data.confidence)
    msg.position.x = data.x
    msg.position.y = data.y
    msg.header.stamp = rospy.Time.from_sec(data.timestamp)
    return msg

def point_msg_from_fixation(fix, raw_data):
    return ibmmpy.msg.FixationDataPoint(
        id = fix.Index,
        start_timestamp = rospy.Time.from_sec(fix.start_timestamp),
        duration = fix.duration,
        x_center = fix.x,
        y_center = fix.y,
        raw_data = [msg_from_gaze_data_point(d) for d in raw_data.itertuples()]
        )
    
## Termination conditions
class EndTime:
    def __init__(self, goal):
        if goal.end_time > rospy.Time.from_sec(0.):
            self.end_time = goal.end_time
        elif goal.detection_time > 0:
            self.end_time = rospy.get_rostime() + rospy.Duration(goal.detection_time)
    def __call__(self, msg, data):
        if hasattr(msg, 'header'): # data message
            return (msg.header.stamp >= self.end_time)
        elif hasattr(msg, 'current_real'): # timer message
            return msg.current_real >= self.end_time
        else:
            return False
    
class NumPoints:
    def __init__(self, goal):
        assert goal.num_points > 0    
        self.num_points = collections.defaultdict(lambda: goal.num_points)
    def __call__(self, msg, data):
        if data is None:
            return False
        done = True
        for k,v in data.items():
            data[k] -= v
            done = done and data[k] <= 0
        return done
                
def get_terminator(goal):
    if goal.end_time > rospy.Time.from_sec(0.) or goal.detection_time > 0:
        return EndTime(goal)
    elif goal.num_points > 0:
        return NumPoints(goal)
    else:
        return lambda m, g: False

## Handle time going backwards
def filter_out_time_backwards(data, prev_time=-np.inf):
    if len(data) == 0:
        return data
    prev_times = np.hstack((prev_time, data.timestamp.values[:-1]))
    filt = data.timestamp.values >= prev_times
    if not all(filt):
        rospy.logwarn('filtered out time going backwards at {} (dt={})'.format(data.iloc[~filt], data.timestamp.values[np.flatnonzero(~filt)]-prev_times[np.flatnonzero(~filt)]))
    return data.iloc[filt,:]


## Action executors
class OnlineCalibratorExecutor:
    def __init__(self, goal):
        self.points = []
        self.detection_criteria = []
        if goal.use_world:
            self.detection_criteria.append('world')
        if goal.use_eye0:
            self.detection_criteria.append('eyes')
        self.model = ibmmpy.ibmm_online.EyeClassifierOnline(dt=goal.label_combination_period, detection_criteria=self.detection_criteria, min_fix_dur=goal.min_fix_duration)
        self.log_dir = goal.log_dir

        
    def callback(self, msg, data):
        self.points.append(data)

    def timer_callback(self, evt):
        pass
                
    def finish(self, parent):
        if len(self.points) == 0:
            return False, 'No data collected'
        data_to_fit = ibmmpy.ibmm_online._call_on_eyes_and_world(lambda l: filter_out_time_backwards(pd.concat(l, ignore_index=True)), 0, self.points)
        num_pts = ibmmpy.ibmm_online._call_on_eyes_and_world(lambda l: len(l[0]), 0, [data_to_fit])
        rospy.loginfo("Got {} messages for calibration ({})".format(len(self.points), str(num_pts)))
        rospy.loginfo("Training model (may take some time to complete)...")
        try:
            self.model.train(data_to_fit)
        except ValueError:
            return False, 'Failed to collect enough valid data for full calibration'
        rospy.loginfo("Training complete.")
        if self.log_dir != "":
            save_calibration_info(self.log_dir, model, data_to_fit)

        parent.model = self.model
        parent.calibration_data = data_to_fit
        rospy.loginfo('Calibration complete')
        return True, ''
        
def load_calibration_from_dir(load_dir):
    with open(os.path.join(load_dir, "model.pkl"), 'r') as f:
        model = pickle.load(f)
    rospy.loginfo("Loaded model {}/model.pkl".format(load_dir))
    # TODO: also load data and re-train to make sure it matches?
    return model

def save_calibration_info(dest, model, data, overwrite=False):
    if os.path.basename(dest) == '' or os.path.isdir(dest):
        # we think it's a directory...
        if not os.path.isdir(dest):
            os.makedirs(dest)
        if not os.path.isdir(dest):
            raise ValueError("Failed to create directory {}".format(dest))

        model_file = os.path.join(dest, 'model.pkl')
        data_file = os.path.join(dest, 'data.pkl')
    else:
        # assume it's a file pattern
        model_file = dest.format('model')
        data_file = dest.format('data')
        if model_file == data_file:
            raise ValueError('Failed to format {}: format generated identical string'.format(dest))
    
    if not overwrite:
        if os.path.exists(model_file):
            raise ValueError('File exists: {}'.format(model_file))
        elif os.path.exists(data_file):
            raise ValueError('File exists: {}'.format(data_file))
    
    with open(model_file, 'w') as f:
        pickle.dump(model, f)
    with open(data_file, 'w') as f:
        pickle.dump(data, f)

    
class DetectorExecutor:
    def __init__(self, current_goal, model, pub):
        self.model = model
        self.model.dt = current_goal.label_combination_period
        self.model.min_fix_dur = current_goal.min_fix_duration
        self.pub = pub
        self._prev_time = {'world': -np.inf, 'eyes': [-np.inf, -np.inf]}

    def callback(self, msg, data):
        data = ibmmpy.ibmm_online._call_on_eyes_and_world(lambda l: filter_out_time_backwards(l[0], l[1]), 0, [data, self._prev_time])
        self._prev_time = ibmmpy.ibmm_online._call_on_eyes_and_world( lambda l: l[0].timestamp.values[-1] if len(l[0]) > 0 else -np.inf, 0, [data] )

        fix, raw_gaze = self.model.classify(data)
        self.publish(fix, raw_gaze, msg.header.stamp)
        cur_time = rospy.get_rostime()
        if (cur_time > msg.header.stamp + rospy.Duration(0.5)):
            rospy.logwarn_throttle(1., 'Processing delay is {:.03f} s'.format( (cur_time - msg.header.stamp).to_sec()  ))

    def timer_callback(self, evt):
        # called when a timer evt is fired without a message
        # finish up the previous fixation so we don't hang on the next data
        fix, raw = self.model.finish()
        self.publish(fix, raw, rospy.get_rostime())
            
    def publish(self, fix, raw_data, tm):
        for f, r in zip(fix.itertuples(), raw_data):
            msg = point_msg_from_fixation(f, r) 
            msg.header.stamp = tm
            self.pub.publish(msg)
            
    def finish(self, parent):
        fix, raw = self.model.finish()
        self.publish(fix, raw, rospy.get_rostime())
        return True, ''
    
# Overall execution
class FixationDetector:
    WATCHDOG_DURATION = rospy.Duration(1.0)
    def __init__(self):
        self.server = actionlib.SimpleActionServer('~detect', ibmmpy.msg.DetectorAction, None, False)
        self.server.register_goal_callback(self.execute)
        self.model = None
        self.calibration_data = None
        self.pub = rospy.Publisher('~fixations', ibmmpy.msg.FixationDataPoint, queue_size=10)
        self.cal_saver = rospy.Service('~save_calibration', ibmmpy.srv.SaveCalibration, self._save_calibration)
        
    def start(self):
        self.server.start()
        self.current_goal = None
        rospy.loginfo('Waiting for goal message')
        
    def calibrate(self, cal_file, cal_goal):
        executor = OnlineCalibratorExecutor(cal_goal)
        with rosbag.Bag(cal_file, 'r') as bag:
            for _, msg, _ in bag.read_messages(topics=[cal_goal.topic]):
                data = gaze_data_from_msg(msg)
                executor.callback(msg, data)
        res, msg = executor.finish(self)
        if not res:
            rospy.logwarn('Failed to perform pre-calibration: {}'.format(msg))
    
    def execute(self):
        if self.current_goal:
            self.finish()
        current_goal = self.server.accept_new_goal()
        rospy.loginfo('Got goal callback, goal: {}'.format(current_goal))

        if self.server.is_preempt_requested():
            rospy.logwarn('Goal preempted immediately after request')
            self.server.set_preempted()
            return
        
        if current_goal.action == ibmmpy.msg.DetectorGoal.ACTION_CALIBRATE:
            if current_goal.load_dir != "":
                try:
                    self.model = load_calibration_from_dir(current_goal.load_dir)
                    self.server.set_succeeded(None, "Loaded calibration from {}".format(current_goal.load_dir))
                except RuntimeError as ex:
                    self.server.set_aborted(None, str(ex))
                return
            else:
                self.executor = OnlineCalibratorExecutor(current_goal)
        elif current_goal.action == ibmmpy.msg.DetectorGoal.ACTION_DETECT:
            if self.model is None:
                self.server.set_aborted(None, 'Must calibrate the detector before running!')
                return
            else:
                self.executor = DetectorExecutor(current_goal, self.model, self.pub)
        else:
            self.server.set_aborted(None, 'Unknown action requested: {}'.format(current_goal.action))
            return

        try:
            self.sub = rospy.Subscriber(current_goal.topic, ibmmpy.msg.GazeData, self._callback)
        except ValueError as ex:
            self.server.set_aborted(None, str(ex))
            return
        
        self.current_goal = current_goal
        self.timer = rospy.Timer(FixationDetector.WATCHDOG_DURATION, self._timer_callback, oneshot=False)
        self.terminator = get_terminator(self.current_goal)
        self.last_active_time = rospy.get_rostime()
        
    def _callback(self, msg):
        data = gaze_data_from_msg(msg)
        self.executor.callback(msg, data)
        if self.server.is_preempt_requested() or self.terminator(msg, data):
            rospy.loginfo('Termination condition reached.')
            self.finish()
        # extend the keepalive timer
        self.last_active_time = rospy.get_rostime()
        
    def _timer_callback(self, msg):
        if self.server.is_preempt_requested() or self.terminator(msg, None):
            rospy.loginfo('Termination condition reached.')
            self.finish()
        elif self.last_active_time is None or (msg.last_real and self.last_active_time <= msg.last_real):
            rospy.logwarn('No gaze data received from {} for at least {} s'.format(self.current_goal.topic, self.timer._period.to_sec()))
            self.executor.timer_callback(msg)

    def _save_calibration(self, req):
        if self.model is None or self.calibration_data is None:
            return ibmmpy.srv.SaveCalibrationResponse(ok=False, msg="Service not calibrated!")
        try:
            save_calibration_info(req.destination, self.model, self.calibration_data, req.overwrite)
        except Exception as ex:
            return ibmmpy.srv.SaveCalibrationResponse(ok=False, msg=str(ex))
        else:
            return ibmmpy.srv.SaveCalibrationResponse(ok=True, msg="Saved to {}".format(req.destination))

            
    def finish(self):
        self.current_goal = None
        # stop callbacks
        self.sub.unregister()
        self.timer.shutdown()
        
        # end executor
        rospy.loginfo("Finishing goal...")
        try:
            res, msg = self.executor.finish(self)
            if res:
                self.server.set_succeeded(None, "Action completed")
            else:
                self.server.set_aborted(None, msg)
        except Exception as e:
            rospy.logerr("Exception when finishing msg:\n{}".format(traceback.format_exc()))
            self.server.set_aborted(None, str(e))
            
        
def main():
    rospy.init_node("fixation_detector")
    detector = FixationDetector()
    offline_cal_file = rospy.get_param('~calibration_bag_file', '')
    offline_cal_dir = rospy.get_param('~calibration_load_dir', '')
    if offline_cal_file != '':
        use_world = rospy.get_param('~calibration_world', False)
        use_eye0 = rospy.get_param('~calibration_eye0', False)
        use_eye1 = rospy.get_param('~calibration_eye1', False)
        topic = rospy.get_param('~calibration_topic', 'gaze')
        log_dir = rospy.get_param('~calibration_log_dir', '')
        goal = ibmmpy.msg.DetectorGoal(topic=topic,
                use_world=use_world, use_eye0=use_eye0, use_eye1=use_eye1, log_dir=log_dir)
        rospy.loginfo('Running offline calibration from {}, goal spec {}'.format(offline_cal_file, goal))
        detector.calibrate(offline_cal_file, goal)
    elif offline_cal_dir != '':
        detector.model = load_calibration_from_dir(offline_cal_dir)
        
    detector.start()
    rospy.spin()
    
if __name__ == '__main__':
    main()
        


