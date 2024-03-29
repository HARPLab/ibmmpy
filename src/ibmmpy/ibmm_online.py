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
    def __init__(self, dt):
        self.dt = dt
        self._prev_raw = []
        self._last_time_cutoff = None
        self._prev_label = None
        
    def __call__(self, raw_labels):
        if self.dt is None or len(raw_labels) == 0 or self.dt == 0:
            return raw_labels.loc[:,['timestamp','label']] # no grouping specified so a no-op
        
        labels = []
        ts = []
        cts = []
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
            fused_label, counts = ibmm.EyeClassifier._fuse_local(selected_raw_labels)
            if fused_label is None:
                if len(labels) > 0: # just copy the previous bc no data given
                    labels.append(labels[-1])
                elif self._prev_label is not None:
                    labels.append(self._prev_label)
                else:
                    labels.append(ibmm.EyeClassifier.LABEL_NOISE)
            else:
                labels.append(fused_label)
            cts.append(counts)
            ts.append(self._last_time_cutoff)
            self._last_time_cutoff += self.dt
        # save the extra bits
        self._prev_raw = raw_labels[raw_labels.timestamp >= self._last_time_cutoff]
        if len(labels) > 0:
            self._prev_label = labels[-1]
        return pd.concat((pd.DataFrame({'timestamp': ts, 'label': labels}), pd.DataFrame(cts)), axis=1)
    
    def reset(self):
        if len(self._prev_raw) > 0:
            fused_labels, cts = ibmm.EyeClassifier._fuse_local(self._prev_raw)
            final_data = pd.concat((
                pd.DataFrame({'timestamp': [np.min(self._prev_raw.timestamp)], 
                    'label': fused_labels if fused_labels is not None else ibmm.EyeClassifier.LABEL_NOISE}),
                pd.DataFrame([cts])), axis=1)
            final_data = final_data.append({'timestamp': final_data.timestamp[0]+self.dt,
                               'label': ibmm.EyeClassifier.LABEL_NOISE}, ignore_index=True)
        elif self._last_time_cutoff is not None:
            final_data = pd.DataFrame([self._last_time_cutoff, ibmm.EyeClassifier.LABEL_NOISE], columns=['timestamp', 'label'])
        else:
            final_data = pd.DataFrame([], columns=['timestamp', 'label'])
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
            if len(self._prev_labels) > 1: # if we've already collected some data
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
                
        self._prev_labels = pd.DataFrame()
        return all_labels.loc[mask,:]
    
class _FixationDetector():
    def __init__(self, min_fix_dur=None, max_fix_dur=np.inf):
        self.min_fix_dur = min_fix_dur
        self.max_fix_dur = max_fix_dur
        self._prev_data = {'world': pd.DataFrame(), 'eyes': [ pd.DataFrame(), pd.DataFrame() ]}
        self._prev_labels = pd.DataFrame()
        self._last_fix_id = -1

    # need to create this so pickle calls __init__ on old-style classes
    def __getinitargs__(self):
        return self.min_fix_dur, self.max_fix_dur
        
    def __call__(self, raw_data, labels):
        data = _call_on_eyes_and_world(lambda lst: pd.concat(lst), 0, (self._prev_data, raw_data))
        if len(labels) == 0:
            # just return if no new data
            self._prev_data = data
            return pd.DataFrame(columns=['start_timestamp', 'duration']), []
        labels = pd.concat((self._prev_labels, labels))
        
        last_idx = np.argmax(labels.timestamp.values)
        if labels.label.values[last_idx] == ibmm.EyeClassifier.LABEL_FIX:
            # we're in the middle of a fixation
            # so trim off the ongoing fixation and save it as previous data
            # we'll update next time we get data
            fix_idx = np.flatnonzero(labels.label != ibmm.EyeClassifier.LABEL_FIX)
            last_fix_idx = fix_idx[-1]+1 if len(fix_idx) > 0 else 0
            
            # if the saved fix data is longer than max_fix_dur, only save past that duration
            dur_to_save = (labels.timestamp.values[-1] - labels.timestamp.values[last_fix_idx])*1e3
            if dur_to_save >= self.max_fix_dur:
                break_tm = labels.timestamp.values[last_fix_idx] + self.max_fix_dur * np.floor(dur_to_save/self.max_fix_dur)*1e-3
                break_idx = labels.timestamp.values.searchsorted(break_tm, side='left')
                break_val = labels.iloc[[break_idx]].copy()
                break_val.timestamp = break_tm

                self._prev_labels = pd.concat((break_val, labels.iloc[break_idx:, :]), axis=0, ignore_index=True)
                self._prev_data = _call_on_eyes_and_world(lambda d: d[0][d[0].timestamp >= break_tm], 0, (data,))

                labels = labels.iloc[:break_idx,:]
                if labels.timestamp.values[-1] < break_tm:
                    labels = pd.concat((labels, break_val), axis=0, ignore_index=True)
            else:
                self._prev_labels = labels.iloc[last_fix_idx:, :]
                self._prev_data = _call_on_eyes_and_world(lambda d: d[0][d[0].timestamp >= labels.timestamp.values[last_fix_idx]], 0, (data,))
                # clear to make sure we don't get a trailing fixation (no need to clear the data here)
                labels = labels.iloc[:last_fix_idx,:]
        else:
            self._prev_labels = labels.tail(1)
            self._prev_data = _call_on_eyes_and_world(lambda d: d[0][d[0].timestamp >= labels.timestamp.values[-1]], 0, (data,))

        fix, gaze_raw = ibmm.EyeClassifier.get_fixations_from_labels(labels, data['world'] if 'world' in data else None, self.min_fix_dur, self.max_fix_dur)
        if len(fix) > 0:
            fix.index = fix.index + self._last_fix_id + 1
            self._last_fix_id = fix.index.values[-1]
        return fix, gaze_raw
    
    def reset(self):
        fix, gaze_raw = ibmm.EyeClassifier.get_fixations_from_labels(self._prev_labels, self._prev_data['world'] if 'world' in self._prev_data else None, self.min_fix_dur, self.max_fix_dur)
        fix.index = fix.index + self._last_fix_id + 1
        self._prev_data = {'world': pd.DataFrame(), 'eyes': [ pd.DataFrame(), pd.DataFrame() ]}
        self._prev_labels = pd.DataFrame()
        self._last_fix_id = -1
        return fix, gaze_raw

        
class EyeClassifierOnline(object):
    def __init__(self, detection_criteria=['world', 'eyes'], dt=None, min_fix_dur=100, max_fix_dur=1000):
        self._classifier = ibmm.EyeClassifier()
        self._preprocess = _Preprocessor(self)
        self._fuse = _LabelFuser(dt)
        self._postprocess = _LabelPostprocessor()
        self._get_fixations = _FixationDetector(min_fix_dur, max_fix_dur)
        self.detection_criteria = detection_criteria
        self._is_running = False

    @property
    def dt(self):
        return self._fuse.dt
    @dt.setter
    def dt(self, dt):
        if self._is_running:
            raise ValueError('Cannot set dt value while ibmmpy is running. Call finish() before setting value.')
        if dt > self.min_fix_dur*1e-3:
            raise ValueError('Must have dt <= min_fix_dur')
        self._fuse.dt = dt
        
    @property
    def min_fix_dur(self):
        return self._get_fixations.min_fix_dur
    @min_fix_dur.setter
    def min_fix_dur(self, min_fix_dur):
        if self._is_running:
            raise ValueError('Cannot set min_fix_dur value while ibmmpy is running. Call finish() before setting value.')
        if min_fix_dur < self.dt*1e3:
            raise ValueError('Must have min_fix_dur >= dt')
        self._get_fixations.min_fix_dur = min_fix_dur
        
    @property
    def max_fix_dur(self):
        return self._get_fixations.max_fix_dur
    @max_fix_dur.setter
    def max_fix_dur(self, max_fix_dur):
        if self._is_running:
            raise ValueError('Cannot set min_fix_dur value while ibmmpy is running. Call finish() before setting value.')
        if max_fix_dur < self.min_fix_dur:
            raise ValueError('Need max_fix_dur >= min_fix_dur')
        self._get_fixations.max_fix_dur = max_fix_dur
        
    def train(self, data):
        # data: a dictionary with {'world': world data, 'eyes': one or two-length list of data}
        data_filt = {k:v for k,v in data.items() if k in self.detection_criteria}
        processed_data = _call_on_eyes_and_world(lambda l: self._classifier.preprocess(l[0]), 0, [data_filt])
        self._classifier.fit(**processed_data)
    
    def classify(self, raw_point):
        self._is_running = True
        data_filt = {k:v for k,v in raw_point.items() if k in self.detection_criteria}
        cur_vel = self._preprocess(data_filt)
#         print('velocity: {}'.format(cur_vel))
        raw_labels = self._classifier.predict(fuse=False, **cur_vel)
#         print('raw labels: {}'.format(raw_labels))
        processed_labels = self._fuse(raw_labels)
#         print('fused labels: {}'.format(processed_labels))
        postprocessed_labels = self._postprocess(processed_labels)
#         print('postprocessed labels: {}'.format(postprocessed_labels))
        fix, gaze_raw = self._get_fixations(raw_point, postprocessed_labels)
        return fix, gaze_raw
    
    def finish(self):
        self._preprocess.reset()
        last_processed = self._fuse.reset()
        last_postprocessed = pd.concat((self._postprocess(last_processed), self._postprocess.reset()), sort=False)
        last_fix1, last_raw1 = self._get_fixations( {'world': pd.DataFrame(), 'eyes': [ pd.DataFrame(), pd.DataFrame() ]}, last_postprocessed)
        last_fix2, last_raw2 = self._get_fixations.reset()
        last_fix = pd.concat((last_fix1, last_fix2), sort=False)
        last_raw1.extend(last_raw2)
        self._is_running = False
        return last_fix, last_raw1
                
                                        

