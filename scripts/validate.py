#!/usr/bin/env python

import ibmmpy
import ibmmpy.test
import numpy as np
import pandas as pd
import os
import argparse
import itertools

def compare_equals(d1, d2, **kwargs):
    return np.allclose(d1.values, d2.loc[:, d1.columns].values)

def compare_intersection_over_union(d1, d2, **kwargs):
    # get sorted intervals
    d1 = d1.iloc[np.argsort(d1.start_timestamp),:]
    d2 = d2.iloc[np.argsort(d2.start_timestamp),:]
    
    intersection = 0
    union = 0
    union_max = -np.inf
    it1 = iter(d1.itertuples())
    it2 = iter(d2.itertuples())
    f1 = next(it1)
    f2 = next(it2)
    
    try:
        while True:
            if f1.start_timestamp > f2.start_timestamp:
                # swap them
                it1, it2 = it2, it1
                f1, f2 = f2, f1
            print(f1)
            print(f2)
            
            if f1.start_timestamp + f1.duration*1e-3 >= f2.start_timestamp:
                intersection += min(f1.start_timestamp + f1.duration*1e-3, f2.start_timestamp + f2.duration*1e-3) - f2.start_timestamp
            union += f1.duration*1e-3 - max(f1.start_timestamp)
            print(intersection)
            print(union)
            f1 = next(it1)
    except StopIteration:
        pass
    union += sum(f.duration*1e-3 for f in it1)
    union += sum(f.duration*1e-3 for f in it2)
    
    return intersection / union

def compare_iou(d1, d2, label_dt, **kwargs):
    # this is a terrible way to do it efficiency-wise but it's much simpler to read so yay
    print('time shape: {}, {}'.format(d1.start_timestamp.shape, d2.start_timestamp.shape))
    all_ts = np.hstack((d1.start_timestamp, d1.start_timestamp+d1.duration*1e-3,
                        d2.start_timestamp, d2.start_timestamp+d2.duration*1e-3)).ravel()
    if label_dt:
        ts = np.arange(np.min(all_ts), np.max(all_ts), label_dt)
    else:
        ts = np.linspace(np.min(all_ts), np.max(all_ts), 5000)
    
    l1 = np.zeros(ts.shape, dtype=np.bool)
    for f in d1.itertuples():
        l1[np.logical_and(ts >= f.start_timestamp, ts < f.start_timestamp+f.duration*1e-3)] = 1
    l2 = np.zeros(ts.shape, dtype=np.bool)
    for f in d2.itertuples():
        l2[np.logical_and(ts >= f.start_timestamp, ts < f.start_timestamp+f.duration*1e-3)] = 1
                
    return float(np.count_nonzero(np.logical_and(l1, l2))) / np.count_nonzero(np.logical_or(l1, l2))
        

def run_offline(data, seed=None, online_dt=None, label_dt=None, data_dir=None, fix_data=['world', 'eyes'], params={}, verbose=False, **kwargs):
    if seed:
        np.random.seed(seed)
    
    class1 = ibmmpy.ibmm.EyeClassifier(**params)
    vs1 = ibmmpy.ibmm_online._call_on_eyes_and_world(lambda d: ibmmpy.ibmm.EyeClassifier.preprocess(d[0]), 0, [data])
    fix_eval_data = {d: vs1[d] for d in fix_data}
    class1.fit(**fix_eval_data)
    fix1 = class1.get_fixations(dt=label_dt, gaze_data=data['world'] if 'world' in data else None, **fix_eval_data)[0]
    return fix1

def run_online(data, seed=None, online_dt=0.1, label_dt=None, data_dir=None, fix_data=['world', 'eyes'], params={}, **kwargs):
    # capture online
    if seed:
        np.random.seed(seed)
    
    class2 = ibmmpy.ibmm_online.EyeClassifierOnline(dt=label_dt, detection_criteria=fix_data, **params)
    class2.train(data)
    
    all_tms = data['world'].timestamp.values.tolist() if 'world' in data else []
    if 'eyes' in data:
        for d in data['eyes']:
            all_tms.extend(d.timestamp.values)
    min_tm = np.min(all_tms)
    max_tm = np.max(all_tms)
    def get_subset_fcn(tmin, tmax):
        def get_subset(data):
            return (data[0])[np.logical_and((data[0]).timestamp >= tmin, (data[0]).timestamp < tmax)]
        return get_subset
    all_fix2 = [class2.classify(ibmmpy.ibmm_online._call_on_eyes_and_world(get_subset_fcn(t, t+online_dt), 0, [data]))[0] for t in np.arange(min_tm, max_tm, online_dt)]
    all_fix2.append(class2.finish()[0])
    print(all_fix2)
    fix2 = pd.concat(all_fix2, sort=False) 
    return fix2

def run_disk(data, data_dir=None, **kwargs):
    if data_dir is None:
        raise ValueError('Must supply data directory to use disk method')
    fix = pd.read_csv(os.path.join(data_dir, 'processed', 'fixations.csv'))
    return pd.DataFrame({'start_timestamp': fix.start_timestamp,
                         'duration': fix.duration,
                         'x': fix.norm_pos_x,
                         'y': fix.norm_pos_x},
                        columns=['start_timestamp', 'duration', 'x', 'y'])
    
def run_bag(data, data_dir=None, bag_file=None, **kwargs):
    if not bag_file and not data_dir:
        raise ValueError('For bag method, must specify either data dir or bag file')
    bag_file = bag_file or os.path.join(data_dir, 'processed', 'fixations.bag')
    import rosbag
    import ibmmpy.msg
    
    fix = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for _, msg, _ in bag.read_messages(topics=['/fixation_detector/fixations']):
            fix.append({'start_timestamp': msg.start_timestamp.to_sec(),
                        'duration': msg.duration,
                        'x': msg.x_center,
                        'y': msg.y_center})
    return pd.DataFrame(fix)

def compare(fix, metrics, params):
    for k1, k2 in itertools.combinations(fix.keys(), 2):
        print('==========\n{} <--> {}:'.format(k1, k2))
        for n, m, in metrics.iteritems():
            print('\t{}: {}'.format(n, m(fix[k1], fix[k2], **params)))

eval_methods = {
    'offline': run_offline,
    'online': run_online,
    'disk': run_disk,
    'bag': run_bag
}

metrics = {
    'equal': compare_equals,
    'iou': compare_iou
    }

def main():
    parser = argparse.ArgumentParser('Testing and validation for ibmmpy online method')
    parser.add_argument('data_dir', nargs='?', default=None, help='Data directory to use (default: use generated test data')
    parser.add_argument('--methods', nargs='+', default=['offline', 'online'], choices=eval_methods.keys(), help='Methods to use')
    parser.add_argument('--metrics', nargs='+', default=['equal'], choices=metrics.keys(), help='Metrics to use')
    parser.add_argument('--seed', default=None, type=int, help='Random seed to use (default: don\'t set)')
    parser.add_argument('--online-dt', default=0.1, type=float, help='Sample bunching rate for online data')
    parser.add_argument('--label-dt', default=None, type=float, help='Sample bunching rate for label voting')
    parser.add_argument('--world-only', default=False, action='store_true', help='use world data only for detection')
    parser.add_argument('--eyes-only', default=False, action='store_true', help='use eye data only for detection')
    parser.add_argument('--verbose', '-v',  default=False, action='store_true', help='print verbose output')
    parser.add_argument('--bag-file', default=None, help='location of bag file with fixations to read (default: processed/fixations.bag)')
    args = parser.parse_args()
    
    if args.data_dir is None:
        data = {
                'world': ibmmpy.test.gaze_data,
                'eyes': [ibmmpy.test.synth_data0, ibmmpy.test.synth_data1]
            }
    else:
        eye0 = pd.read_csv(os.path.join(args.data_dir, 'text_data', 'pupil_eye0.csv'))
        eye1 = pd.read_csv(os.path.join(args.data_dir, 'text_data', 'pupil_eye1.csv'))
        gaze = pd.read_csv(os.path.join(args.data_dir, 'text_data', 'gaze_positions.csv'))
        data_eye0 = pd.DataFrame({'timestamp': eye0.timestamp,
                                  'confidence': eye0.confidence,
                                  'x': eye0.circle_3d_normal_x,
                                  'y': eye0.circle_3d_normal_y,
                                  'z': eye0.circle_3d_normal_z
                                  })
        data_eye1 = pd.DataFrame({'timestamp': eye1.timestamp,
                                  'confidence': eye1.confidence,
                                  'x': eye1.circle_3d_normal_x,
                                  'y': eye1.circle_3d_normal_y,
                                  'z': eye1.circle_3d_normal_z
                                  })
        
        gaze_data = pd.DataFrame({'timestamp': gaze.timestamp,
                                  'confidence': gaze.confidence,
                                  'x': gaze.norm_pos_x,
                                  'y': gaze.norm_pos_y})
        data = {'world': gaze_data, 'eyes': [data_eye0, data_eye1]}
    
    
    params = {'seed': args.seed,
                   'online_dt': args.online_dt,
                   'label_dt': args.label_dt,
                   'data_dir': args.data_dir,
                   'verbose': args.verbose,
                   'bag_file': args.bag_file}
    if args.world_only:
        params['fix_data'] = ['world']
    elif args.eyes_only:
        params['fix_data'] = ['eyes']
    else:
        params['fix_data'] = ['world', 'eyes']
    
    all_fix = { method: eval_methods[method](data, **params) for method in args.methods }
    ref_time = all_fix.values()[0].start_timestamp[0]
    for k, v in all_fix.items():
        print('==== {} =====\n{}'.format(k, v.assign(start_timestamp=v.start_timestamp-ref_time)))
    
    compare(all_fix, {n: metrics[n] for n in args.metrics}, params)
    
if __name__ == '__main__':
#     a = pd.DataFrame([[0, 100.], [1, 100.], [2, 100.], [3, 100]], columns=['start_timestamp', 'duration'])
#     b =  pd.DataFrame([[0.05, 100]], columns=['start_timestamp', 'duration'])
#     print(compare_intersection_over_union(a, b))
    main()
    



