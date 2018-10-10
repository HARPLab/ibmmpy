#!/usr/bin/env python

from ibmm import EyeClassifier
import pandas as pd
import numpy as np

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

if __name__ == "__main__":
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
    
    
    
