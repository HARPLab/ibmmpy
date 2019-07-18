#!/usr/bin/env python

import ibmm
import pandas as pd
import numpy as np

def _call_on_eyes_and_world(func, test_id, lst):
    out = {}
    if 'world' in lst[test_id]:
        out['world'] = func([l['world'] for l in lst])
    if 'eyes' in lst[test_id]:
        if len(lst[test_id]['eyes']) > 0:
            out['eyes'] = [func([l['eyes'][0] for l in lst])]
        if len(lst[test_id]['eyes']) > 1:
            out['eyes'].append(func([l['eyes'][1] for l in lst]))
    return out
    

class EyeClassifierOnline:
    def __init__(self, detection_criteria=['world', 'eyes'], dt=None, min_fix_dur=100):
        self._classifier = ibmm.EyeClassifier()
        self._preprocess = EyeClassifierOnline._Preprocessor(self)
        self._fuse = EyeClassifierOnline._LabelFuser(dt)
        self._postprocess = EyeClassifierOnline._LabelPostprocessor()
        self._get_fixations = EyeClassifierOnline._FixationDetector(min_fix_dur)
        self.detection_criteria = detection_criteria
        
    class _Preprocessor():
        def __init__(self, parent):
            self._parent = parent
            self._prev_raw = {'world': pd.DataFrame(), 'eyes': [ pd.DataFrame(), pd.DataFrame() ]}
        
        def __call__(self, raw_point):
            all_data = _call_on_eyes_and_world(lambda lst: pd.concat(lst), 1, (self._prev_raw, raw_point))
            cur_vel = _call_on_eyes_and_world(lambda l: self._parent._classifier.preprocess(l[0]), 0, [all_data])
            
            def update_last_row_if_used_data(vel, prev_row):
                if len(prev_row) > 0:
                    return vel.tail(len(vel)-1)
                else:
                    return vel
            cur_vel = _call_on_eyes_and_world(lambda l: update_last_row_if_used_data(*l), 0, (cur_vel, self._prev_raw))            
            
            def get_last_row(l):
                if len(l) > 0:
                    return l.tail(1)
                else:
                    return pd.DataFrame()
            self._prev_raw = _call_on_eyes_and_world(lambda l: l[0].tail(1) if len(l[0]) > 0 else pd.DataFrame(), 0, [all_data])          
            return cur_vel
        
        def reset(self):
            self._prev_raw = {'world': pd.DataFrame(), 'eyes': [ pd.DataFrame(), pd.DataFrame() ]}
        
    class _LabelFuser():
        def __init__(self, dt, ):
            self.dt = dt
            self._prev_raw = []
            self._last_time_cutoff = None
            self._prev_label = None
            
        def __call__(self, raw_labels):
            if self.dt is None or len(raw_labels) == 0:
                return raw_labels.loc[:,['timestamp','label']] # no grouping specified so a no-op
            
            labels = []
            ts = []
            if self._last_time_cutoff is None:
                self._last_time_cutoff = np.min(raw_labels.timestamp)
            else:
                # add in the previous extra labels
                raw_labels = pd.concat((self._prev_raw, raw_labels))
            max_time = np.max(raw_labels.timestamp)
            while self._last_time_cutoff + self.dt < max_time:
                tprev = self._last_time_cutoff
                tnext = self._last_time_cutoff + self.dt
                selected_raw_labels = raw_labels[np.logical_and(raw_labels.timestamp >= tprev, raw_labels.timestamp < tnext)]
                fused_label, _ = ibmm.EyeClassifier._fuse_local(selected_raw_labels)
                if fused_label is None:
                    if len(labels) > 0: # just copy the previous bc no data given
                        labels.append(labels[-1])
                    elif self._prev_label is not None:
                        labels.append(self._prev_label)
                    else:
                        labels.append(ibmm.EyeClassifier.LABEL_NOISE)
                else:
                    labels.append(fused_label)
                self._last_time_cutoff += self.dt
                ts.append(self._last_time_cutoff)
            # save the extra bits
            self._prev_raw = raw_labels[raw_labels.timestamp >= self._last_time_cutoff]
            if len(labels) > 0:
                self._prev_label = labels[-1]
            
            return pd.DataFrame({'timestamp': ts, 'label': labels})
        
        def reset(self):
            if len(self._prev_raw) > 0:
                fused_labels, _ = ibmm.EyeClassifier._fuse_local(self._prev_raw)
                final_data = pd.DataFrame({'timestamp': [np.max(self._prev_raw.timestamp)], 'label': fused_labels if fused_labels is not None else ibmm.EyeClassifier.LABEL_NOISE})
            else:
                final_data = pd.DataFrame()
            self._prev_raw = []
            self._last_time_cutoff = None
            return final_data
            
        
    class _LabelPostprocessor():
        def __init__(self):
            self._prev_labels = pd.DataFrame()
        
        def __call__(self, labels):
            all_labels = pd.concat((self._prev_labels, labels), ignore_index=True, sort=False)
            fixed_labels = ibmm.EyeClassifier.postprocess(all_labels.label.values)
            all_labels.label = fixed_labels
            mask = np.full(len(all_labels), True, dtype=bool)
            if len(all_labels) > 0: # make sure we've actually started collecting data
                if len(self._prev_labels) > 0: # if we've already collected some data
                    # don't double-send the first point since we sent it last time
                    mask[0] = False
                    
                self._prev_labels = all_labels.tail(2)
                # strip off the last point since we haven't confirmed it yet
                mask[-1] = False
            return all_labels.loc[mask,:]
        
        def reset(self):
            all_labels = self._prev_labels.copy()
            mask = np.full(len(all_labels), True, dtype=bool)
            if len(all_labels) > 0: # make sure we've actually started collecting data
                if len(self._prev_labels) > 0: # if we've already collected some data
                    # don't double-send the first point since we sent it last time
                    mask[0] = False
                    
                # strip off the last point since we haven't confirmed it yet
            self._prev_labels = pd.DataFrame()
            return all_labels.loc[mask,:]
        
    class _FixationDetector():
        def __init__(self,  min_fix_dur):
            self.min_fix_dur = min_fix_dur
            self._prev_data = {'world': pd.DataFrame(), 'eyes': [ pd.DataFrame(), pd.DataFrame() ]}
            self._prev_labels = pd.DataFrame()
            self._last_fix_id = -1
            
        def __call__(self, raw_data, labels):
            data = _call_on_eyes_and_world(lambda lst: pd.concat(lst), 0, (self._prev_data, raw_data))
            
            if len(labels) == 0:
                # just return if no new data
                self._prev_data = data
                return pd.DataFrame(columns=['start_timestamp', 'duration'])
            labels = pd.concat((self._prev_labels, labels))
            
            last_idx = np.argmax(labels.timestamp.values)
            if labels.label.values[last_idx] == ibmm.EyeClassifier.LABEL_FIX:
                # we're in the middle of a fixation
                # so trim off the ongoing fixation and save it as previous data
                # we'll update next time we get data
                fix_idx = np.flatnonzero(labels.label != ibmm.EyeClassifier.LABEL_FIX)
                last_fix_idx = fix_idx[-1] if np.any(fix_idx) else 0
                self._prev_labels = labels.iloc[last_fix_idx:, :]
                self._prev_data = _call_on_eyes_and_world(lambda d: d[0][d[0].timestamp >= labels.timestamp.values[last_fix_idx]], 0, (data,))
                # clear to make sure we don't get a trailing fixation (no need to clear the data here)
                labels = labels.iloc[:last_fix_idx,:]
            else:
                self._prev_labels = labels.tail(1)
                self._prev_data = _call_on_eyes_and_world(lambda d: d[0][d[0].timestamp >= labels.timestamp.values[-1]], 0, (data,))
                
            fix = ibmm.EyeClassifier.get_fixations_from_labels(labels, data['world'] if 'world' in data else None, self.min_fix_dur)
            if len(fix) > 0:
                fix.index = fix.index + self._last_fix_id + 1
                self._last_fix_id = fix.index.values[-1]
            return fix
        
        def reset(self):
            fix = ibmm.EyeClassifier.get_fixations_from_labels(self._prev_labels, self._prev_data['world'] if 'world' in self._prev_data else None, self.min_fix_dur)
            fix.index = fix.index + self._last_fix_id + 1
            self._prev_data = {'world': pd.DataFrame(), 'eyes': [ pd.DataFrame(), pd.DataFrame() ]}
            self._prev_labels = pd.DataFrame()
            self._last_fix_id = -1
            return fix
                
        
    def train(self, data):
        # data: a dictionary with {'world': world data, 'eyes': one or two-length list of data}
        processed_data = _call_on_eyes_and_world(lambda l: self._classifier.preprocess(l[0]), 0, [data])
        self._classifier.fit(**processed_data)
    
    def classify(self, raw_point):
        cur_vel = self._preprocess(raw_point)
        cur_vel_filt = {k:v for k,v in cur_vel.items() if k in self.detection_criteria}
        raw_labels = self._classifier.predict(fuse=False, **cur_vel_filt)
        processed_labels = self._fuse(raw_labels)
        postprocessed_labels = self._postprocess(processed_labels)
        fix = self._get_fixations(raw_point, postprocessed_labels)
        return fix
    
    def finish(self):
        self._preprocess.reset()
        last_processed = self._fuse.reset()
        last_postprocessed = pd.concat((self._postprocess(last_processed), self._postprocess.reset()), sort=False)
        last_fix = pd.concat((self._get_fixations( {'world': pd.DataFrame(), 'eyes': [ pd.DataFrame(), pd.DataFrame() ]}, last_postprocessed), self._get_fixations.reset()), sort=False)
        return last_fix
                
                                        