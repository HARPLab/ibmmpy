#!/usr/bin/env python

from ibmm import EyeClassifier
from ibmm_online import EyeClassifierOnline
import pandas as pd
import numpy as np

np.random.seed(10)

synth_data0 = pd.DataFrame({
        'timestamp': np.arange(100., 101., 0.05) + np.random.rand( 20 )*0.01,
        'confidence': np.random.rand(20)*0.1 + 0.9,
        'x': np.array([ 0., 0., 0.001, 0., -0.0003, 0.0001, 0.0003, 0.00004, -0.00011, -0.00009, 0.025, 0.051, 0.078, 0.10, 0.1001, 0.101, 0.0998, 0.101, 0.1002, 0.0998 ]),
        'y': np.array([ 0., 0., -0.0002, 0.0003, -0.003, -0.0002, 0.0, 0.0001, 0.0003, 0.00009, 0.015, 0.032, 0.065, 0.07, 0.0702, 0.0678, 0.0698, 0.0702, 0.07012, 0.06998 ])
    }) 


synth_data1 = pd.DataFrame({
        'timestamp': np.arange(100., 101., 0.05) + np.random.rand( 20 )*0.01,
        'confidence': np.concatenate( ( [0.1], np.random.rand(19)*0.1 + 0.9) ),
        'x': np.array([ 0., 0., -0.0002, 0.0003, -0.003, -0.0002, 0.0, 0.0001, 0.0003, 0.00009, 0.02, 0.032, 0.065, 0.07, 0.0702, 0.0678, 0.0698, 0.0702, 0.07012, 0.06998 ]) + np.random.rand(20)*0.001,
        'y': np.array([ 0., 0., 0.001, 0., -0.0003, 0.0001, 0.0003, 0.00004, -0.00011, -0.00009, 0.025, 0.051, 0.078, 0.10, 0.1001, 0.101, 0.0998, 0.101, 0.1002, 0.0998 ]) + np.random.rand(20)*0.001
    })

gaze_data = pd.DataFrame({
        'timestamp': np.arange(100, 101, 0.02),
        'x': np.concatenate(( np.random.rand(25)*.01 + 2., np.random.rand(7)*.01+3, np.random.rand(18)*.01+4 )),
        'y': np.concatenate(( np.random.rand(25)*.01 + 6., np.random.rand(7)*.01+5, np.random.rand(18)*.01+4 ))
    }) 
gaze_data = gaze_data.assign(confidence = np.interp(gaze_data.timestamp, synth_data0.timestamp, synth_data0.confidence))

def test1():
    print(synth_data0)
    print(synth_data1)
    
    vel0 = EyeClassifier.preprocess(synth_data0)
    vel1 = EyeClassifier.preprocess(synth_data1)
    print(vel0)
    print(vel1)
    
    model = EyeClassifier()
    model.fit(eyes=(vel0, vel1))
    
    labels, indiv_labels = model.predict(eyes=(vel0, vel1), ts=np.arange(100., 101., 0.05))
    print(labels)
    
    fix = model.get_fixations(eyes=(vel0, vel1), gaze_data=gaze_data, dt=0.01)
    print(fix)
    
    # world-based
    vel_w = EyeClassifier.preprocess(gaze_data)
    model = EyeClassifier()
    model.fit(world=vel_w)
    fix = model.get_fixations(world=vel_w, gaze_data=gaze_data)
    print(fix)
    
def test2():
    online_processor = EyeClassifierOnline(dt=0.05, detection_criteria=['eyes'])
    online_processor.train({'eyes': (synth_data0, synth_data1)})
    
    dt = 0.01
    fix = []
    for t in np.arange(100, 101, dt):
        subset = {
            'world': gaze_data[np.logical_and(gaze_data.timestamp >= t, gaze_data.timestamp < t+dt)],
            'eyes': (synth_data0[np.logical_and(synth_data0.timestamp >= t, synth_data0.timestamp < t+dt)],
                     synth_data1[np.logical_and(synth_data1.timestamp >= t, synth_data1.timestamp < t+dt)])
            }
        print(subset)
        next_fix = online_processor.classify(subset)
        print('--------\n{}:\n{}'.format(t, next_fix))
        fix.append(next_fix)
    last_fix = online_processor.finish()
    print('--------\n{}:\n{}'.format('last', last_fix))
    fix.append(last_fix)
    
    print('|||||||||||||||\n{}'.format(pd.concat(fix, sort=False)))
    
    
    vel0 = EyeClassifier.preprocess(synth_data0)
    vel1 = EyeClassifier.preprocess(synth_data1)
    fix2 = online_processor._classifier.get_fixations(eyes=(vel0, vel1), gaze_data=gaze_data, dt=0.05)
    print('|||||||||||||\nOffline:\n{}'.format(fix2))
    
    
def randomized_online_test(n=1):
    for i in range(n):
        np.random.seed(i)
        
        # generate fixations 


def test_max_fix_dur():    
    # world-based
    vel_w = EyeClassifier.preprocess(gaze_data)
    model = EyeClassifier()
    model.fit(world=vel_w)

    fix, _ = model.get_fixations(world=vel_w, gaze_data=gaze_data, min_fix_dur=50)
    print(fix)
    
    fix, _ = model.get_fixations(world=vel_w, gaze_data=gaze_data, min_fix_dur=50, max_fix_dur=225)
    print(fix)
    
def test_max_fix_dur_online():
    online_processor = EyeClassifierOnline(dt=0.05, detection_criteria=['world'], max_fix_dur=175, min_fix_dur=30)
    online_processor.train({'world': gaze_data})
    
    def run_test(transmit_dt, label_dt, min_fix_dur, max_fix_dur):
        try:
            online_processor.dt = label_dt
            online_processor.min_fix_dur = min_fix_dur
            online_processor.max_fix_dur = max_fix_dur
        except ValueError:
            print("Skipped: transmit={}, label={}, min={}, max={}".format(transmit_dt, label_dt, min_fix_dur, max_fix_dur))
            return
        fix = []
        for t in np.arange(100, 101, transmit_dt):
            subset = {
                'world': gaze_data[np.logical_and(gaze_data.timestamp >= t, gaze_data.timestamp < t+transmit_dt)],
                'eyes': (pd.DataFrame([], columns=['timestamp', 'confidence', 'x', 'y']),
                    pd.DataFrame([], columns=['timestamp', 'confidence', 'x', 'y']))
                }
            next_fix, _ = online_processor.classify(subset)
            if len(next_fix) > 0:
                fix.append(next_fix)
        last_fix, _ = online_processor.finish()
        fix.append(last_fix)
        online_fix = pd.concat(fix, sort=False)
        offline_fix = online_processor._classifier.get_fixations(world=EyeClassifier.preprocess(gaze_data), 
                gaze_data=gaze_data, min_fix_dur=online_processor.min_fix_dur, max_fix_dur=online_processor.max_fix_dur,
                dt=online_processor.dt)[0]
        
        try:
            if np.allclose(online_fix, offline_fix):
                print("Succeeded: transmit={}, label={}, min={}, max={}".format(transmit_dt, label_dt, min_fix_dur, max_fix_dur))
            else:
                print("Failed: transmit={}, label={}, min={}, max={}".format(transmit_dt, label_dt, min_fix_dur, max_fix_dur))
                print(online_fix)
                print(offline_fix)
        except ValueError:
            print("Failed: transmit={}, label={}, min={}, max={}".format(transmit_dt, label_dt, min_fix_dur, max_fix_dur))
            print(online_fix)
            print(offline_fix)


    for transmit_dt in np.linspace(0.01, 0.1, 4):
        for label_dt in np.linspace(0.01, 0.2, 6):
            for min_fix_dur in np.arange(25, 175, 50):
                for max_fix_dur in np.arange(175, 500, 100):
                    run_test(transmit_dt, label_dt, min_fix_dur, max_fix_dur)
    
if __name__ == "__main__":
    test_max_fix_dur_online()
        
        
        
        
        
    
