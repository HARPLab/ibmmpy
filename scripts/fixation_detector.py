#!/usr/bin/env python

import rospy
import actionlib
import pandas as pd
import ibmmpy.msg
import ibmmpy.ibmm_online
import collections
from ibmmpy.test import gaze_data

## Message generating/parsing
def gaze_data_point_from_msg(msg):
    return {'x': msg.x, 'y': msg.y, 'confidence': msg.confidence, 'timestamp': msg.header.stamp.to_secs()}

def gaze_data_from_msg(msg):
    return {
        'world': pd.DataFrame([gaze_data_from_msg(m) for m in msg.world_data]),
        'eyes': [pd.DataFrame([gaze_data_from_msg(m) for m in msg.eye0_data]),
                 pd.DataFrame([gaze_data_from_msg(m) for m in msg.eye1_data])]
        }
    
def point_msg_from_fixation(fix):
    return ibmmpy.msg.FixationDataPoint(
        id = fix.id,
        start_time = rospy.Time.from_sec(fix.start_timestamp),
        duration = fix.duration,
        x_center = fix.x,
        y_center = fix.y
        )
    
def msg_from_fixations(fix):
    return ibmmpy.msg.FixationData(fixations = [point_msg_from_fixation(f) for f in fix.itertuples()])

## Termination conditions
class EndTime:
    def __init__(self, goal):
        assert goal.end_time > rospy.Time.from_sec(0.)
        self.end_time = goal.end_time
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
    if goal.end_time > rospy.Time.from_sec(0.):
        return EndTime(goal)
    elif goal.num_points > 0:
        return NumPoints(goal)
    else:
        return lambda g: False

## Action executors
class CalibratorExecutor:
    def __init__(self, goal):
        self.points = collections.defaultdict(list)
        self.detection_criteria = []
        if goal.use_world:
            self.detection_criteria.append('world')
        if goal.use_eye0:
            self.detection_criteria.append('eyes')
        self.use_eye1 = goal.use_eye1
        self.model = ibmmpy.ibmm_online.EyeClassifierOnline(dt=goal.label_combination_period, detection_criteria=self.detection_criteria, min_fix_dur=goal.min_fix_duration*1e3)

        
    def callback(self, msg, data):
        if 'world' in self.detection_criteria:
            self.points['world'].extend(data['world'])
        if 'eyes' in self.detection_criteria:
            self.points['eyes'][0].extend(data['eyes'][0])
            if self.use_eye1:
                self.points['eyes'][1].extend(data['eyes'][1])
                
    def finish(self, parent):
        data_to_fit = {k: pd.concat(v) for k, v in self.points.items()}
        self.model.train(data_to_fit)
        parent.model = self.model
    
class DetectorExecutor:
    def __init__(self, model, pub):
        self.model = model
        self.pub = pub
        
    def callback(self, msg, data):
        fix = self.model.classify(data)
        for f in fix.itertuples():
            msg = point_msg_from_fixation(f) 
            msg.header.stamp = data.header.stamp
            self.pub.publish(msg)
            
    def finish(self, parent):
        pass # just need to destruct the callback which happens up a level
    
# Overall execution
class FixationDetector:
    WATCHDOG_DURATION = rospy.Duration(1.0)
    def __init__(self):
        self.server = actionlib.SimpleActionServer('detect', ibmmpy.msg.DetectorAction, self.execute, False)
        self.model = None
        self.pub = rospy.Publisher('fixations', ibmmpy.msg.FixationDataPoint, queue_size=10)
        self.server.start()
    
    def execute(self, goal):
        if self.server.is_preempt_requested():
            self.server.set_aborted(None, 'Preempted')
            return
        self.current_goal = goal
        if goal.action == ibmmpy.msg.DetectorGoal.ACTION_CALIBRATE:
            self.executor = CalibratorExecutor(goal)
        elif goal.action == ibmmpy.msg.DetectorGoal.ACTION_DETECT:
            if self.model is None:
                self.server.set_aborted(None, 'Must calibrate the detector before calibration')
                return
            else:
                self.executor = DetectorExecutor(self.model, self.pub)
        else:
            self.server.set_aborted(None, 'Unknown action requested: {}'.format(goal.action))
            return
        
        self.sub = rospy.Subscriber(goal.topic, ibmmpy.msg.GazeData, self._callback)
        self.timer = rospy.Timer(FixationDetector.WATCHDOG_DURATION, self._timer_callback, oneshot=False)
        self.terminator = get_terminator(goal)
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
        if self.last_active_time <= msg.last_real:
            rospy.logwarn('No gaze data received from {} for at least {}'.format(self.goal.topic, msg.last_duration))
            
    def finish(self):
        # stop callbacks
        self.sub.unregister()
        self.timer.shutdown()
        
        # end executor
        self.executor.finish(self)
        del self.executor
        
        # finish goal
        self.server.set_succeeded(None, "Action terminated")
        
def main():
    rospy.init_node("fixation_detector")
    detector = FixationDetector()
    rospy.spin()
    
if __name__ == '__main__':
    main()
        


