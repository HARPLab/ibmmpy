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
    return pd.DataFrame({
        'id': msg.id,
        'start_timestamp': msg.start_timestamp.to_sec(),
        'duration': msg.duration,
        'x_center': msg.x_center,
        'y_center': msg.y_center
        }, index=[0]
        )
    

class FixationVisualizer:
    def __init__(self, 
                 image_topic='/image',
                 image_subtopic='image_raw',
                 fixation_topic='/fixations',
                 gaze_topic='/gaze',
                 gaze_color=(255, 255, 0), gaze_radius=5, 
                 fix_color=(255, 0, 0), fix_radius=15, linger_time=5):
        self.gaze_color = gaze_color
        self.gaze_radius = gaze_radius
        
        self.fix_color = fix_color
        self.fix_radius_scale = fix_radius # pixel radius / sec 
        
        self.linger_time = linger_time # sec    
        self.cam_info = None
        
        self.fixations = pd.DataFrame()
        self.raw_data = pd.DataFrame()
        
        self.bridge = cv_bridge.CvBridge()
        self.cam_info_sub = rospy.Subscriber(image_topic + '/camera_info', sensor_msgs.msg.CameraInfo, self._cam_info_callback)
        self.image_sub = rospy.Subscriber(image_topic + '/' + image_subtopic, sensor_msgs.msg.Image, self._frame_callback)
        self.fix_sub = rospy.Subscriber(fixation_topic, ibmmpy.msg.FixationDataPoint, self._fix_callback)
        self.gaze_sub = rospy.Subscriber(gaze_topic, ibmmpy.msg.GazeData, self._gaze_callback)
        self.image_pub = rospy.Publisher('~image_overlay', sensor_msgs.msg.Image, queue_size=1)
        
    def draw_fixation(self, frame, fix, cur_time):
        time_since_start = cur_time - fix.start_timestamp
        if time_since_start > fix.duration * 1e-3 + self.linger_time:
            return False
        if np.isnan(fix.x_center) or np.isnan(fix.y_center):
            return False
        
        radius = int(self.fix_radius_scale * min(time_since_start, fix.duration * 1e-3))
        rospy.loginfo('[{}] time dif: {}, dur: {}, radius: {}'.format(fix.Index, time_since_start, fix.duration, radius))
        if radius > 0:
            pos = (int(fix.x_center * self.cam_info.width), int( (1-fix.y_center) * self.cam_info.height ) )
            cv2.circle(frame, pos, radius, self.fix_color, -1)
        return True
    
    def draw_raw_gaze(self, frame, raw_data, cur_time):
        time_since_start = cur_time - raw_data.timestamp
        if time_since_start > self.linger_time:
            return False
        if np.isnan(raw_data.x) or np.isnan(raw_data.y):
            return False
        
        pos = (int(raw_data.x * self.cam_info.width), int( (1-raw_data.y) * self.cam_info.height ) )
        cv2.circle(frame, pos, self.gaze_radius, self.gaze_color, -1)
        return True
        
    
    def draw_frame(self, frame, cur_time):
        cur_num_fix = len(self.fixations) # handle if an addl fix is added while we're running
        fix_alive = [ self.draw_fixation(frame, f, cur_time) for f in self.fixations[:cur_num_fix].itertuples() ]
        self.fixations[:cur_num_fix] = (self.fixations[:cur_num_fix])[fix_alive]
        
        cur_num_gaze = len(self.raw_data)
        gaze_alive = [ self.draw_raw_gaze(frame, r, cur_time) for r in self.raw_data[:cur_num_gaze].itertuples() ]
        self.raw_data[:cur_num_gaze] = (self.raw_data[:cur_num_gaze])[gaze_alive]
        
        return frame
    
    def _fix_callback(self, msg):
        self.fixations = pd.concat((self.fixations, get_fixation_from_point_msg(msg)), ignore_index=True)
    
    def _gaze_callback(self, msg):
        print(msg)
        self.raw_data = pd.concat((self.raw_data, fixation_detector.gaze_data_from_msg(msg)['world']), ignore_index=True)
    
    def _cam_info_callback(self, msg):
        self.cam_info = msg
        self.cam_info_sub.unregister()
        
    def _frame_callback(self, frame):
        if self.cam_info is None:
            rospy.logwarn('No camera info received yet')
            return
        new_frame = self.draw_frame(self.bridge.imgmsg_to_cv2(frame, desired_encoding='bgr8'), frame.header.stamp.to_sec())
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(new_frame, encoding='bgr8'))

def main():
    image_topic = rospy.resolve_name('raw_image')
    rospy.loginfo('reading image from topic {}'.format(image_topic))
    image_subtopic = rospy.get_param('~image_subtopic', 'image_raw')
    fixation_topic = rospy.get_param('~fixation_topic', '/fixations')
    gaze_topic = rospy.get_param('~gaze_topic', '/gaze')
    gaze_color = tuple(int(x) for x in rospy.get_param('~gaze_color', [255, 255, 0]))
    gaze_radius = int(rospy.get_param('~gaze_radius', 5))
    fix_color = tuple(int(x) for x in rospy.get_param('~fixation_color', [255, 0, 0]))
    fix_radius = int(rospy.get_param('~fixation_radius', 15))
    linger_time = rospy.get_param('~linger_time', 5)
    
    viz = FixationVisualizer(image_topic=image_topic, image_subtopic=image_subtopic,
                             fixation_topic=fixation_topic, gaze_topic=gaze_topic,
                             gaze_color=gaze_color, gaze_radius=gaze_radius,
                             fix_color=fix_color, fix_radius=fix_radius,
                             linger_time=linger_time)
    
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node('fixation_visualizer')
    main()
        
        
    