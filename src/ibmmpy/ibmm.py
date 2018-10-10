#!/usr/bin/env python

import sklearn.mixture
import sklearn.metrics.pairwise
import numpy as np
import pandas as pd

class EyeClassifier:
    LABEL_FIX = 0
    LABEL_SAC = 1
    LABEL_NOISE = -1 
    
    def __init__(self, **kwargs):
        
        self.eye_models = (sklearn.mixture.BayesianGaussianMixture(n_components=2, weight_concentration_prior_type='dirichlet_distribution', **kwargs),
                           sklearn.mixture.BayesianGaussianMixture(n_components=2, weight_concentration_prior_type='dirichlet_distribution', **kwargs))
        self.eye_labels = None
        self.world_model = sklearn.mixture.BayesianGaussianMixture(n_components=2, weight_concentration_prior_type='dirichlet_distribution', **kwargs)
        self.world_labels = None
    
    @staticmethod
    def preprocess(data, dist_method='vector', conf_thresh=0.8, smoothing='none'):
        """
        Preprocess input x/y positions to get pairwise distances and remove low-confidence values
        
        Keyword arguments:
        data -- Input pandas-style dataframe with columns 'timestamp', 'confidence', 'x', 'y', optionally 'z' (ignored if method is 'euclidean', used if method is 'vector' and set to 1.0 if missing)
        dist_method -- 'vector' (compute angle between eye rays) or 'euclidean' (compute euclidean distance)
        conf_threshold -- valid confidence to accept
        smoothing -- preprocessing of signal to smooth values, options currently only include 'none'
        
        Returns:
        a pandas dataframe with columns 'timestamp', 'velocity' (which may be nan, corresponding to noise)
        """
        # Validate arguments
        DIST_METHOD_VECTOR = 'vector'
        DIST_METHOD_EUC = 'euclidean'
        DIST_METHODS = [DIST_METHOD_VECTOR, DIST_METHOD_EUC]
        if dist_method not in DIST_METHODS:
            raise ValueError('Unrecognized distance method {}, must be one of {}'.format(dist_method, DIST_METHODS))
        
        SMOOTHING_NONE = 'none'
        SMOOTHINGS = [SMOOTHING_NONE]
        if smoothing not in SMOOTHINGS:
            raise ValueError('Unrecognized smoothing method {}, must be one of {}'.format(smoothing, SMOOTHINGS))
        
        # Data extraction
        if 'z' in data.columns and dist_method == DIST_METHOD_VECTOR:
            pos = data.loc[:, ['x','y','z']].values
        elif dist_method == DIST_METHOD_VECTOR:
            pos = np.hstack( (data.loc[:, ['x','y']].values, np.ones( (len(data), 1) )) )
        elif dist_method == DIST_METHOD_EUC:
            pos = data.loc[:, ['x','y']]
        else:
            raise RuntimeError("unreachable")
        
        # TODO: smoothing
        
        
        # Compute velocity
        if dist_method == DIST_METHOD_VECTOR:
            sdist = sklearn.metrics.pairwise.paired_cosine_distances(pos[1:,:], pos[:-1,:])
            # Clamp to handle numeric errors
            sdist[sdist > 1.] = 1.
            sdist[sdist < -1.] = -1.
            dist = np.arcsin(sdist)
        elif dist_method == DIST_METHOD_EUC:
            dist = sklearn.metrics.pairwise.paired_euclidean_distances(pos[1:,:], pos[:-1,:])
        else:
            raise RuntimeError("unreachable")
        dt = np.diff(data['timestamp'].values)
        
        veloc = dist / dt
        
        # Remove low-confidence values
        veloc[ np.logical_or(data.confidence.values[1:] < conf_thresh,
                             data.confidence.values[:-1] < conf_thresh) ] = np.nan
        # add a nan value at the beginning so the data point count remains the same
        veloc = np.concatenate( ([np.nan], veloc) )
        
        return pd.DataFrame({'timestamp': data['timestamp'], 'velocity': veloc}, index=data.index)
    
    @staticmethod
    def _fit(model, data):
        model.fit(data.loc[np.logical_not(np.isnan(data['velocity'])), 'velocity'].values.reshape(-1,1))
        
    def fit(self, eyes=None, world=None):
        """
        Fit the bayesian mixture models for each eye.
                
        Arguments:
        eyes -- An iterable of length 1 or 2, including preprocessed data (in the format output by preprocess() ), of eye positions
        world -- Preprocessed world positions
        """
        if eyes is not None:
            if len(eyes) > 0:
                EyeClassifier._fit(self.eye_models[0], eyes[0])
                if self.eye_models[0].means_[0] < self.eye_models[0].means_[1]:
                    self.eye_labels = [np.array([EyeClassifier.LABEL_FIX, EyeClassifier.LABEL_SAC])]
                else:
                    self.eye_labels = [np.array([EyeClassifier.LABEL_SAC, EyeClassifier.LABEL_FIX])]
            if len(eyes) > 1:
                EyeClassifier._fit(self.eye_models[1], eyes[1])
                if self.eye_models[1].means_[0] < self.eye_models[1].means_[1]:
                    self.eye_labels.append(np.array([EyeClassifier.LABEL_FIX, EyeClassifier.LABEL_SAC]))
                else:
                    self.eye_labels.append(np.array([EyeClassifier.LABEL_SAC, EyeClassifier.LABEL_FIX]))
    
        if world is not None:
            EyeClassifier._fit(self.world_model, world)
            if self.world_model.means_[0] < self.world_model.means_[1]:
                self.world_labels = [np.array([EyeClassifier.LABEL_FIX, EyeClassifier.LABEL_SAC])]
            else:
                self.world_labels = [np.array([EyeClassifier.LABEL_SAC, EyeClassifier.LABEL_FIX])]
            
    @staticmethod
    def _predict(model, model_labels, data):
        labels = np.ones(len(data))*EyeClassifier.LABEL_NOISE
        valid_mask = np.logical_not(np.isnan(data['velocity']))
        labels[valid_mask] = model_labels[model.predict(data.loc[valid_mask, 'velocity'].values.reshape(-1,1))]
        return labels
    
    @staticmethod
    def postprocess(labels, noise_only=True):
        """
        Post-process label assignments to clean up noise related stuff.
        
        For now, all this does is finds sequences ABA where B != A, A != noise, and converts them to AAA.
        
        This could be more sophisticated someday.
        """
        if noise_only:
            labels_to_fix = np.logical_and(
                np.logical_and(labels[0:-2] == labels[2:], labels[1:-1] == EyeClassifier.LABEL_NOISE),
                labels[0:-2] != EyeClassifier.LABEL_NOISE)
        else:
            labels_to_fix = np.logical_and(
                np.logical_and(labels[0:-2] == labels[2:], labels[1:-1] != labels[0:-2] ),
                labels[0:-2] != EyeClassifier.LABEL_NOISE)
        indices = np.flatnonzero(labels_to_fix)
        labels[indices+1] = labels[indices]
        return labels
        
    @staticmethod
    def fuse(labels, ts=None, dt=None):
        """
        Fuse different label sets to come to an agreement.
        
        Algorithm, roughly:
            For each period n*dt - (n+1)*dt:
                Find labels from each set within the period
                Result label = majority vote among SAC, FIX; NSE if all are noise; break ties as SAC
        
        Arguments:
        labels -- Iterable of pandas-style dataframe with columns 'timestamp' and 'label'
        ts -- list of timestamps to sample at, or None to use automatic samples from dt
        dt -- sampling period to use, if ts is None
        
        Returns:
        pandas DataFrame with columns:
            timestamp -- ts or times generated from dt
            label -- the fused labels
        """
        if ts is None:
            ts = np.arange( min(l.timestamp.values[0] for l in labels), max(l.timestamp.values[-1] for l in labels), dt)
        fused_labels = np.zeros(ts.shape, dtype=np.int8) + EyeClassifier.LABEL_NOISE
        cts_sac = np.zeros(ts.shape, dtype=np.int8)
        cts_fix = np.zeros(ts.shape, dtype=np.int8)
        cts_nse = np.zeros(ts.shape, dtype=np.int8)
        for idx in range(ts.size):
            tprev = ts[idx-1] if idx > 0 else -np.inf
            tnext = ts[idx]
            cur_labels = np.concatenate( [l.label[np.logical_and(l.timestamp > tprev, l.timestamp <= tnext)] for l in labels] )
            ct_sac = np.count_nonzero(cur_labels == EyeClassifier.LABEL_SAC)
            ct_fix = np.count_nonzero(cur_labels == EyeClassifier.LABEL_FIX)
            
            cts_sac[idx] = ct_sac
            cts_fix[idx] = ct_fix
            cts_nse[idx] = np.count_nonzero(cur_labels == EyeClassifier.LABEL_NOISE)
            
            if ct_sac > 0:
                fused_labels[idx] = EyeClassifier.LABEL_SAC
            elif ct_fix > 0:
                fused_labels[idx] = EyeClassifier.LABEL_FIX
#             if ct_fix > ct_sac:
#                 fused_labels[idx] = EyeClassifier.LABEL_FIX
#             elif ct_sac > ct_fix or ct_sac > 0:# they're equal and it's not entirely noise
#                 fused_labels[idx] = EyeClassifier.LABEL_SAC
#             elif ct_fix == 0 and ct_sac == 0 and idx > 0: # there's no data available
                # leave it as noise for now
#                 fused_labels[idx] = fused_labels[idx-1]
            # otherwise leave as noise
            
        # Fix length-one holes
        fused_labels = EyeClassifier.postprocess(fused_labels)
        
        data = pd.DataFrame({'timestamp': ts, 'label': fused_labels,
                             'count_fix': cts_fix,
                             'count_sac': cts_sac,
                             'count_nse': cts_nse})
        data.index.name = 'id'
        return data
        
    
    def predict(self, eyes=None, world=None, ts=None, dt=None):
        """
        Predict labels from a collection of data.
        
        First predicts the labels for each type of data from the internal model. Then fuses the data, as described in fuse().
        
        Arguments:
        eyes -- 1- or 2-length iterable of eye data, in the format output by preprocess() above, or None
        world -- Eye data in the world frame, in the format output by preprocess() above, or None
        ts -- Timestamps to use, or None. See fuse() for logic.
        dt -- Time period to use, or None. See fuse() for logic.
        
        Returns:
        Tuple of (fused_data, iterable of labels found)
        """
        data_to_fuse = []
        if eyes is not None:
            if len(eyes) > 0:
                labels0 = EyeClassifier._predict(self.eye_models[0], self.eye_labels[0], eyes[0])
                labels0 = EyeClassifier.postprocess(labels0)
                data_to_fuse.append(pd.DataFrame({'timestamp': eyes[0].timestamp, 'label': labels0}))
            if len(eyes) > 1:
                labels1 = EyeClassifier._predict(self.eye_models[1], self.eye_labels[1], eyes[1])
                labels1 = EyeClassifier.postprocess(labels1)
                data_to_fuse.append(pd.DataFrame({'timestamp': eyes[1].timestamp, 'label': labels1}))

        if world is not None:
            labelsw = EyeClassifier._predict(self.world_model, self.world_labels, world)
            labelsw = EyeClassifier.postprocess(labelsw)
            data_to_fuse.append(pd.DataFrame({'timestamp': world.timestamp, 'label': labelsw}))
        return EyeClassifier.fuse(data_to_fuse, ts, dt), data_to_fuse
        
    @staticmethod
    def get_fixations_from_labels(labels, gaze_data=None, min_fix_dur=100):
        """
        Convert a sequence of labels into detected fixations.
        
        Arguments:
        labels -- pandas-style dataframe with columns 'timestamp', 'label', as in supplied by EyeClassifier.fuse
        gaze_data -- Pandas-style dataframe with columns 'timestamp', 'x', 'y', or None. Fills out fixation 'x', 'y' if provided
        min_fix_dur -- minimum fixation duration to filter out (in ms), or None if no filtering is to be done
        
        Returns:
        pandas dataframe of all detected fixations with columns 'start_timestamp', 'duration' (in ms). If gaze_data is provided, also
        includes columns 'x', 'y', which are the mean of the values of gaze_data.x and gaze_data.y for the duration of the fixation
        """

        # Fixations are periods of either "fixation" or "noise" that start and end with a fixation label
        # possibly there should be a limit to the amount of noise allowed within a fixation?
#         is_fix = np.logical_or(labels.label.values == EyeClassifier.LABEL_FIX,
#                                 labels.label.values == EyeClassifier.LABEL_NOISE).astype(np.int8)

        is_fix = (labels.label.values == EyeClassifier.LABEL_FIX).astype(np.int8)
        fix_change = is_fix[1:] - is_fix[:-1]

        fix_start = np.flatnonzero(fix_change == 1) + 1
        fix_end = np.flatnonzero(fix_change == -1)
        if is_fix[0]:
            if is_fix[1]:
                # if there's a length-2 to start, make sure to mark it
                fix_start = np.concatenate( ([0], fix_start) )
            else:
                # if it's just length 1, remove it
                fix_end = fix_end[1:]
        if is_fix[-1]:
            if is_fix[-2]:
                fix_end = np.concatenate( (fix_end, [is_fix.size-1]))
            else:
                fix_start = fix_start[:-1]

        # Shrink fixation start and end periods to reject noise at the edges
#         def get_offset_idx(st,nd):
#             nse_idx = np.flatnonzero( labels.label.values[st:nd] != EyeClassifier.LABEL_NOISE )
#             if nse_idx.size > 0:
#                 return nse_idx[ [0,-1] ]
#             else:
#                 return [ nd-st, nd-st ]
#         noise_offsets = np.array([ get_offset_idx(st,nd) for st,nd in zip(fix_start, fix_end) ])
#         fix_end -= fix_end - fix_start - noise_offsets[:,1]
#         fix_start += noise_offsets[:,0]
        # now make sure we didn't overshoot (possible only if a "fixation" is entirely noise"
        ok_idx = fix_start < fix_end
        fix_start = fix_start[ok_idx]
        fix_end = fix_end[ok_idx]
        

        fix = pd.DataFrame({ 'start_timestamp': labels.timestamp.values[fix_start], 'duration': (labels.timestamp.values[fix_end] - labels.timestamp.values[fix_start]) * 1000. })


        #Filter out too-short fixations
        if min_fix_dur is not None:
            fix = fix.loc[fix.duration >= min_fix_dur, :]
            fix.index = np.arange(len(fix))
        if gaze_data is not None:
            m_x = [ np.mean( gaze_data.x[np.logical_and(gaze_data.timestamp.values >= r.start_timestamp,
                                                                       gaze_data.timestamp.values <= r.start_timestamp + .001*r.duration)] )
                   for r in fix.itertuples() ]
            m_y = [ np.mean( gaze_data.y[np.logical_and(gaze_data.timestamp.values >= r.start_timestamp,
                                                                       gaze_data.timestamp.values <= r.start_timestamp + .001*r.duration)] )
                   for r in fix.itertuples() ]
            fix = fix.assign(x=m_x, y=m_y)
        fix.index.name = 'id'
        return fix
    
    def get_fixations(self, eyes=None, world=None, ts=None, dt=None, gaze_data=None, min_fix_dur=100):
        if ts is None and dt is None and gaze_data is not None:
            ts = gaze_data.timestamp.values
        labels, _ = self.predict(eyes=eyes, world=world, ts=ts, dt=dt)
        print(labels)
        return EyeClassifier.get_fixations_from_labels(labels, gaze_data, min_fix_dur)
            
        
        

