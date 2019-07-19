#!/usr/bin/env python

import rospy
import rosbag
import numpy as np
import pandas as pd
import os
import argparse
import ibmmpy.msg
from geometry_msgs.msg import Point


def gaze_data_point_to_msg(data, selector):
    msg = ibmmpy.msg.GazeDataPoint(
        confidence=data.confidence,
        position=Point(*selector(data)))
    msg.header.stamp = rospy.Time.from_sec(data.timestamp)
    return msg

def select_world(data):
    return (data.norm_pos_x, data.norm_pos_y, np.nan)

def select_eye(data):
    return (data.circle_3d_normal_x, data.circle_3d_normal_y, data.circle_3d_normal_z)
    
def gaze_data_to_msg(world, eye0, eye1, tm):
    msg = ibmmpy.msg.GazeData(
            world_data=[gaze_data_point_to_msg(d, select_world) for d in world.itertuples()],
            eye0_data=[gaze_data_point_to_msg(d, select_eye) for d in eye0.itertuples()],
            eye1_data=[gaze_data_point_to_msg(d, select_eye) for d in eye1.itertuples()],
        )
    msg.header.stamp = tm
    return msg
    
def main():
    parser = argparse.ArgumentParser(description="convert gaze data to bag for processing")
    parser.add_argument('data_dir', help="data directory to process")
    parser.add_argument('--topic', default='gaze', help="output topic for gaze data")
    parser.add_argument('--period', default=0.033, type=float, help="sample period for grouping data")
    args = parser.parse_args()
    
    world_data = pd.read_csv(os.path.join(args.data_dir, 'text_data', 'gaze_positions.csv'))
    eye0_data = pd.read_csv(os.path.join(args.data_dir, 'text_data', 'pupil_eye0.csv'))
    eye1_data = pd.read_csv(os.path.join(args.data_dir, 'text_data', 'pupil_eye1.csv'))
    with rosbag.Bag(os.path.join(args.data_dir, 'processed', 'fixations.bag'), 'w') as bag:
        all_ts = np.hstack((world_data.timestamp, eye0_data.timestamp, eye1_data.timestamp))
        max_tm = np.max(all_ts)
        tms = np.arange(np.min(all_ts), max_tm, args.period)
        if tms[-1] < max_tm:
            tms = np.hstack((tms, tms[-1]+args.period))
        
        for t0, t1 in zip(tms[:-1], tms[1:]):
            bag.write(
                topic=args.topic,
                t=rospy.Time.from_sec(t1),
                msg=gaze_data_to_msg(
                        world_data[np.logical_and(world_data.timestamp >= t0, world_data.timestamp < t1)],
                        eye0_data[np.logical_and(eye0_data.timestamp >= t0, eye0_data.timestamp < t1)],
                        eye1_data[np.logical_and(eye1_data.timestamp >= t0, eye1_data.timestamp < t1)],
                        rospy.Time.from_sec(t1)
                    )
                )
            
if __name__ == "__main__":
    main()
        
    
