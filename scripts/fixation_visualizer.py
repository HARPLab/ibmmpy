#!/usr/bin/env python
import cv2
import pandas as pd
import numpy as np
import cv_bridge
import rospy
import sensor_msgs.msg
import ibmmpy.msg
import fixation_detector


def get_fixation_from_point_msg(msg):
    if not isinstance(msg.id, int):
        print('unexpected msg type: {}'.format(type(msg.id)))
        print(msg)
        raise Exception
    return pd.DataFrame({
        'id': msg.id,
        'start_timestamp': msg.start_timestamp.to_sec(),
        'duration': msg.duration,
        'x_center': msg.x_center,
        'y_center': msg.y_center
        }, index=[msg.id]
        ).dropna()
    

class FixationVisualizer:
    __COLORS__ = ( (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255) )
    __UNUSED_COLOR__ = ( 192, 192, 192 )
    __GAZE_RADIUS__ = 5
    __FIX_RADIUS__ = 15
    __LINGER_COUNT__ = len(__COLORS__)
    __MAX_PUB_PERIOD__ = rospy.Duration(0.1)

    def __init__(self, 
                 image_topic='/image',
                 image_subtopic='image_raw',
                 fixation_topic='/fixations',
                 gaze_topic='/gaze'):
        
        self.cam_info = None
        
        self.fixations = pd.DataFrame()
        self.raw_data = pd.DataFrame()
        
        self.bridge = cv_bridge.CvBridge()

        if image_topic is not None:
            self.cam_info_sub = rospy.Subscriber(image_topic + '/camera_info', sensor_msgs.msg.CameraInfo, self._cam_info_callback)
            self.image_sub = rospy.Subscriber(image_topic + '/' + image_subtopic, sensor_msgs.msg.Image, self._frame_callback)
            self._last_image = None
        else:
            self.cam_info = sensor_msgs.msg.CameraInfo(height=768, width=1024)
            self.timer = rospy.Timer(rospy.Duration.from_sec(0.1), self._timer_callback, oneshot=False)
            self._last_image = np.zeros((self.cam_info.height, self.cam_info.width, 3), dtype=np.uint8)
        
        self.fix_sub = rospy.Subscriber(fixation_topic, ibmmpy.msg.FixationDataPoint, self._fix_callback)
        self.gaze_sub = rospy.Subscriber(gaze_topic, ibmmpy.msg.GazeData, self._gaze_callback)
        self.image_pub = rospy.Publisher('~image_overlay', sensor_msgs.msg.Image, queue_size=1)

        self._last_time = None
        self._last_pub_time = None
        
    def draw_fixations(self, frame, fixations):
        for fix in fixations.itertuples():
            pos = (int(fix.x_center * self.cam_info.width), int( (1-fix.y_center) * self.cam_info.height ) )
            color = FixationVisualizer.__COLORS__[ fix.id % len(FixationVisualizer.__COLORS__) ]
            cv2.circle(frame, pos, FixationVisualizer.__FIX_RADIUS__, color, 3)
            
    def draw_raw_gaze(self, frame, raw_data, fixations):

        fix_iter = fixations.itertuples()
        try:
            fix = next(fix_iter)
        except StopIteration:
            fix, fix_iter = None, None

        for gaze in raw_data.itertuples():
            while fix_iter is not None and fix.start_timestamp + 1e-3*fix.duration < gaze.timestamp:
                try:
                    fix = next(fix_iter)
                except StopIteration:
                    fix, fix_iter = None, None
            if fix is None or gaze.timestamp < fix.start_timestamp:
                color = FixationVisualizer.__UNUSED_COLOR__
            else:
                color = FixationVisualizer.__COLORS__[fix.id % len(FixationVisualizer.__COLORS__)]
            pos = (int(gaze.x * self.cam_info.width), int( (1-gaze.y) * self.cam_info.height ) )
            cv2.circle(frame, pos, FixationVisualizer.__GAZE_RADIUS__, color, -1)
        
    
    def draw_frame(self, frame, cur_time):
        # handle if an addl fix is added while we're running
        fix = self.fixations
        self.draw_fixations(frame, fix) 
        
        raw_gaze = self.raw_data
        self.draw_raw_gaze(frame, raw_gaze, fix)
        
        return frame

    def compile_data_and_publish(self, tm):
        # call this from all the callbacks
        # basically bc if we're running in sim time, and pause the bag, the timer callback pauses
        # but the data is still coming in so problems
        if self._last_image is None:
            return
        if self._last_pub_time is not None and (tm - self._last_pub_time) < FixationVisualizer.__MAX_PUB_PERIOD__:
            return
        self._last_pub_time = tm
        frame = self._last_image.copy()
        new_frame = self.draw_frame(frame, tm.to_sec())
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(new_frame, encoding='bgr8'))
    
    def _fix_callback(self, msg):
        self.fixations = pd.concat((self.fixations, get_fixation_from_point_msg(msg)))
        if len(self.fixations) > FixationVisualizer.__LINGER_COUNT__:
            self._last_time = self.fixations.iloc[-FixationVisualizer.__LINGER_COUNT__-1].start_timestamp + self.fixations.iloc[-FixationVisualizer.__LINGER_COUNT__-1].duration*1e-3
            self.fixations = self.fixations[-FixationVisualizer.__LINGER_COUNT__:]
        self.compile_data_and_publish(msg.header.stamp)
    
    def _gaze_callback(self, msg):
        new_data = pd.DataFrame(fixation_detector.gaze_data_from_msg(msg)['world']).dropna().set_index('timestamp', drop=False)
        if len(new_data) > 0 and len(self.raw_data) > 0:
            filt = np.hstack((self.raw_data.iloc[-1].timestamp, new_data.iloc[:-1].timestamp.values)) <= new_data.timestamp.values
            if not np.all(filt):
                rospy.logwarn('new data went backwards: {}'.format(new_data.iloc[~filt]))
                new_data = new_data.iloc[filt,:]
        self.raw_data = pd.concat((self.raw_data, new_data))
        if not self.raw_data.index.is_monotonic:
            rospy.logwarn(''.join('new data not monotonic: ',
                self.raw_data.assign(tm=lambda r: r.timestamp-msg.header.stamp.to_sec()).tail(),
                new_data.assign(tm=lambda r: r.timestamp-msg.header.stamp.to_sec())))
        _last_time = self._last_time
        if _last_time is not None:
            self.raw_data = self.raw_data.truncate(before=_last_time)
        
        self.compile_data_and_publish(msg.header.stamp)
    
    def _cam_info_callback(self, msg):
        self.cam_info = msg
        self.cam_info_sub.unregister()
        
    def _frame_callback(self, frame):
        if self.cam_info is None:
            rospy.logwarn('No camera info received yet')
            return
        self._last_image = self.bridge.imgmsg_to_cv2(frame, desired_encoding='bgr8')
        self.compile_data_and_publish(frame.header.stamp)
        
    def _timer_callback(self, event):
        rospy.logdebug('got timer callback at {}'.format(event.current_real))
        self.compile_data_and_publish(event.current_real)


def main():
    if rospy.get_param('~overlay'):
        image_topic = rospy.resolve_name('raw_image')
        image_subtopic = rospy.get_param('~image_subtopic', 'image_raw')
        rospy.loginfo('reading image from topic {}'.format(image_topic))
    else:
        image_topic = None
        image_subtopic = None
        rospy.loginfo('generating empty fixation image')
    fixation_topic = rospy.get_param('~fixation_topic', '/fixations')
    gaze_topic = rospy.get_param('~gaze_topic', '/gaze')
    # gaze_color = tuple(int(x) for x in rospy.get_param('~gaze_color', [255, 255, 0]))
    # gaze_radius = int(rospy.get_param('~gaze_radius', 5))
    # fix_color = tuple(int(x) for x in rospy.get_param('~fixation_color', [255, 0, 0]))
    # fix_radius = int(rospy.get_param('~fixation_radius', 15))
    # linger_time = rospy.get_param('~linger_time', 5)
    
    viz = FixationVisualizer(image_topic=image_topic, image_subtopic=image_subtopic,
                             fixation_topic=fixation_topic, gaze_topic=gaze_topic)
    
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node('fixation_visualizer')
    main()
        
        
    