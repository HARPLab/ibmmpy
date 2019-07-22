#!/usr/bin/env python

import rospy
import actionlib
import pandas as pd
import numpy as np
import ibmmpy.msg
import ibmmpy.ibmm_online
import collections
import rosbag

## Message generating/parsing
def gaze_data_point_from_msg(msg):
    data = {'x': msg.position.x, 'y': msg.position.y, 'confidence': msg.confidence, 'timestamp': msg.header.stamp.to_sec()}
    if not np.isnan(msg.position.z):
        data['z'] = msg.position.z
    return data

def gaze_data_from_msg(msg):
    return {
        'world': pd.DataFrame([gaze_data_point_from_msg(m) for m in msg.world_data]),
        'eyes': [pd.DataFrame([gaze_data_point_from_msg(m) for m in msg.eye0_data]),
                 pd.DataFrame([gaze_data_point_from_msg(m) for m in msg.eye1_data])]
        }
    
def point_msg_from_fixation(fix):
    return ibmmpy.msg.FixationDataPoint(
        id = fix.Index,
        start_timestamp = rospy.Time.from_sec(fix.start_timestamp),
        duration = fix.duration,
        x_center = fix.x,
        y_center = fix.y
        )
    
def msg_from_fixations(fix):
    return ibmmpy.msg.FixationData(fixations = [point_msg_from_fixation(f) for f in fix.itertuples()])

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

## Action executors
class CalibratorExecutor:
    def __init__(self, goal):
        self.points = []
        self.detection_criteria = []
        if goal.use_world:
            self.detection_criteria.append('world')
        if goal.use_eye0:
            self.detection_criteria.append('eyes')
        self.model = ibmmpy.ibmm_online.EyeClassifierOnline(dt=goal.label_combination_period, detection_criteria=self.detection_criteria, min_fix_dur=goal.min_fix_duration*1e3)

        
    def callback(self, msg, data):
        self.points.append(data)
                
    def finish(self, parent):
        if len(self.points) == 0:
            return False, 'No data collected'
        data_to_fit = ibmmpy.ibmm_online._call_on_eyes_and_world(lambda l: pd.concat(l, ignore_index=True), 0, self.points)
        try:
            self.model.train(data_to_fit)
            parent.model = self.model
            rospy.loginfo('Calibration complete')
            return True, ''
        except ValueError:
            return False, 'Failed to collect enough valid data for full calibration'
        
    
class DetectorExecutor:
    def __init__(self, model, pub):
        self.model = model
        self.pub = pub
        
    def callback(self, msg, data):
        fix = self.model.classify(data)
        self.publish(fix, msg.header.stamp)
        cur_time = rospy.get_rostime()
        if (cur_time > msg.header.stamp + rospy.Duration(0.5)):
            rospy.logwarn('Processing delay is {}'.format(cur_time - msg.header.stamp))
            
    def publish(self, fix, tm):
        for f in fix.itertuples():
            msg = point_msg_from_fixation(f) 
            msg.header.stamp = tm
            self.pub.publish(msg)
            
    def finish(self, parent):
        fix = self.model.finish()
        self.publish(fix, rospy.get_rostime())
        return True, ''
    
# Overall execution
class FixationDetector:
    WATCHDOG_DURATION = rospy.Duration(1.0)
    def __init__(self):
        self.server = actionlib.SimpleActionServer('detect', ibmmpy.msg.DetectorAction, None, False)
        self.server.register_goal_callback(self.execute)
        self.model = None
        self.pub = rospy.Publisher('fixations', ibmmpy.msg.FixationDataPoint, queue_size=10)
        self.server.start()
        self.current_goal = None
        
    def calibrate(self, cal_file, cal_goal):
        executor = CalibratorExecutor(cal_goal)
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
            self.executor = CalibratorExecutor(current_goal)
        elif current_goal.action == ibmmpy.msg.DetectorGoal.ACTION_DETECT:
            if self.model is None:
                self.server.set_aborted(None, 'Must calibrate the detector before calibration')
                return
            else:
                self.executor = DetectorExecutor(self.model, self.pub)
        else:
            self.server.set_aborted(None, 'Unknown action requested: {}'.format(current_goal.action))
            return
        
        self.current_goal = current_goal
        self.sub = rospy.Subscriber(self.current_goal.topic, ibmmpy.msg.GazeData, self._callback)
        self.timer = rospy.Timer(FixationDetector.WATCHDOG_DURATION, self._timer_callback, oneshot=False)
        self.terminator = get_terminator(self.current_goal)
        self.last_active_time = rospy.get_rostime()
    
    def _callback(self, msg):
        data = gaze_data_from_msg(msg)
        self.executor.callback(msg, data)
        if self.server.is_preempt_requested() or self.terminator(msg, data):
            self.finish()
        # extend the keepalive timer
        self.last_active_time = rospy.get_rostime()
        
    def _timer_callback(self, msg):
        if self.server.is_preempt_requested() or self.terminator(msg, None):
            self.finish()
        elif self.last_active_time is None or (msg.last_real and self.last_active_time <= msg.last_real):
            rospy.logwarn('No gaze data received from {} for at least {} s'.format(self.current_goal.topic, self.timer._period.to_sec()))
            
    def finish(self):
        self.current_goal = None
        # stop callbacks
        self.sub.unregister()
        self.timer.shutdown()
        
        # end executor
        try:
            res, msg = self.executor.finish(self)
            if res:
                self.server.set_succeeded(None, "Action completed")
            else:
                self.server.set_aborted(None, msg)
        except Exception as e:
            self.server.set_aborted(None, e.message)
            
        
def main():
    rospy.init_node("fixation_detector")
    detector = FixationDetector()
    offline_cal_file = rospy.get_param('~calibration_file', '')
    if offline_cal_file != '':
        use_world = rospy.get_param('~calibration_world', False)
        use_eye0 = rospy.get_param('~calibration_eye0', False)
        use_eye1 = rospy.get_param('~calibration_eye1', False)
        topic = rospy.get_param('~calibration_topic', 'gaze')
        goal = ibmmpy.msg.DetectorGoal(topic=topic,
                use_world=use_world, use_eye0=use_eye0, use_eye1=use_eye1)
        rospy.loginfo('Running offline calibration from {}, goal spec {}'.format(offline_cal_file, goal))
        detector.calibrate(offline_cal_file, goal)
    
    rospy.spin()
    
if __name__ == '__main__':
    main()
        


