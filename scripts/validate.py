#!/usr/bin/env python

import ibmmpy
import ibmmpy.test
import numpy as np
import pandas as pd
import os
import argparse

def compare_equals(d1, d2):
    return d1.equals(d2)

def compare_online_offline(data, seed=None, online_dt=0.1, label_dt=None, metric=compare_equals, params={}):
    # capture offline
    if seed:
        np.random.seed(seed)
    
    class1 = ibmmpy.ibmm.EyeClassifier(**params)
    vs1 = ibmmpy.ibmm_online._call_on_eyes_and_world(lambda d: ibmmpy.ibmm.EyeClassifier.preprocess(d[0]), 0, [data])
    class1.fit(**vs1)
    fix1 = class1.get_fixations(dt=label_dt, gaze_data=data['world'] if 'world' in data else None, **vs1)[0]
    
    # capture online
    if seed:
        np.random.seed(seed)
    
    class2 = ibmmpy.ibmm_online.EyeClassifierOnline(dt=label_dt, **params)
    class2.train(data)
    
    all_tms = np.hstack([data['world'].timestamp if 'world' in data else [], np.ravel(np.array([d.timestamp for d in data['eyes']])) if 'eyes' in data else []])
    min_tm = np.min(all_tms)
    max_tm = np.max(all_tms)
    def get_subset_fcn(tmin, tmax):
        def get_subset(data):
            return (data[0])[np.logical_and((data[0]).timestamp >= tmin, (data[0]).timestamp < tmax)]
        return get_subset
    all_fix2 = [class2.classify(ibmmpy.ibmm_online._call_on_eyes_and_world(get_subset_fcn(t, t+online_dt), 0, [data]))[0] for t in np.arange(min_tm, max_tm, online_dt)]
    all_fix2.append(class2.finish())
    fix2 = pd.concat(all_fix2, sort=False)
    
    # compare
    return metric(fix1, fix2), fix1, fix2

def main():
    parser = argparse.ArgumentParser('Testing and validation for ibmmpy online method')
    parser.add_argument('data_dir', nargs='?', default=None, help='Data directory to use (default: use generated test data')
    parser.add_argument('--seed', default=None, type=int, help='Random seed to use (default: don\'t set)')
    parser.add_argument('--online-dt', default=0.1, type=float, help='Sample bunching rate for online data')
    parser.add_argument('--label-dt', default=None, type=float, help='Sample bunching rate for label voting')
    args = parser.parse_args()
    
    if args.data_dir is None:
        data = {
                'world': ibmmpy.test.gaze_data,
                'eyes': [ibmmpy.test.synth_data0, ibmmpy.test.synth_data1]
            }
    else:
        pass # todo: import data
    
    res, f1, f2 = compare_online_offline(data, args.seed, args.online_dt, args.label_dt)
    print('Result: {}'.format(res))
    print('== offline ========= \n{}\n== online ==========\n{}'.format(f1, f2))
    
if __name__ == '__main__':
    main()
    



