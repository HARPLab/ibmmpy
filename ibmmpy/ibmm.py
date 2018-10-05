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
        
        self.bmm_eye0 = sklearn.mixture.BayesianGaussianMixture(n_components=2, weight_concentration_prior_type='dirichlet_distribution', **kwargs)
        self.bmm_eye1 = sklearn.mixture.BayesianGaussianMixture(n_components=2, weight_concentration_prior_type='dirichlet_distribution', **kwargs)
    
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
        
        # Remove low-confidence values 
        pos[data['confidence'] < conf_thresh] = np.nan
        
        # Compute velocity
        if dist_method == DIST_METHOD_VECTOR:
            dist = np.arcsin(sklearn.metrics.pairwise.paired_cosine_distances(pos[1:,:], pos[:-1,:]))
        elif dist_method == DIST_METHOD_EUC:
            dist = sklearn.metrics.pairwise.paired_euclidean_distances(pos[1:,:], pos[:-1,:])
        else:
            raise RuntimeError("unreachable")
        dt = np.diff(data['timestamp'].values)
        # add a nan value at the beginning so the data point count remains the same
        veloc = np.concatenate( ([np.nan], dist / dt) )
        return pd.DataFrame({'timestamp': data['timestamp'], 'velocity': veloc}, index=data.index)
    
    @staticmethod
    def _fit(model, data):
        model.fit(data.loc[np.logical_not(np.isnan(data['velocity'])), 'velocity'].values.reshape(-1,1))
        
    def fit(self, data0, data1):
        """
        Fit the bayesian mixture models for each eye.
        
        Arguments:
        data0 -- Preprocessed data for eye 0 (in the format output by preprocess() above)
        data1 -- Preprocessed data for eye 1
        """
        EyeClassifier._fit(self.bmm_eye0, data0)
        if self.bmm_eye0.means_[0] < self.bmm_eye0.means_[1]:
            self.eye0_labels = np.array([EyeClassifier.LABEL_FIX, EyeClassifier.LABEL_SAC])
        else:
            self.eye0_labels = np.array([EyeClassifier.LABEL_SAC, EyeClassifier.LABEL_FIX])
        EyeClassifier._fit(self.bmm_eye1, data1)
        if self.bmm_eye1.means_[0] < self.bmm_eye1.means_[1]:
            self.eye1_labels = np.array([EyeClassifier.LABEL_FIX, EyeClassifier.LABEL_SAC])
        else:
            self.eye1_labels = np.array([EyeClassifier.LABEL_SAC, EyeClassifier.LABEL_FIX])
    
    @staticmethod
    def _predict(model, model_labels, data):
        labels = np.ones(len(data))*EyeClassifier.LABEL_NOISE
        valid_mask = np.logical_not(np.isnan(data['velocity']))
        labels[valid_mask] = model_labels[model.predict(data.loc[valid_mask, 'velocity'].values.reshape(-1,1))]
        return labels
    
    @staticmethod
    def postprocess(labels):
        """
        Post-process label assignments to clean up noise related stuff.
        
        For now, all this does is finds sequences ABA where B != A, A != noise, and converts them to AAA.
        
        This could be more sophisticated someday.
        """
        labels_to_fix = np.logical_and(
                np.logical_and(labels[0:-2] == labels[2:], labels[0:-2] != labels[1:-1]),
                labels[0:-2] != EyeClassifier.LABEL_NOISE)
        indices = np.nonzero(labels_to_fix)[0]
        labels[indices+1] = labels[indices]
        return labels
        
    @staticmethod
    def fuse(labels0, labels1, ts=None, dt=None):
        """
        Fuse two different label sets to come to an agreement.
        
        Algorithm, roughly:
            For each period n*dt - (n+1)*dt:
                Find labels from each set within the period
                Result label = majority vote among SAC, FIX; NSE if all are noise; break ties as SAC
        
        Arguments:
        labels0 -- pandas-style dataframe with columns 'timestamp' and 'label', as determined by EyeClassifier.postprocess
        labels1 -- second set as in labels0, e.g. from the other eye
        ts -- list of timestamps to sample at, or None to use automatic samples from dt
        dt -- sampling period to use, if ts is None
        
        Returns:
        pandas DataFrame with columns:
            timestamp -- ts or times generated from dt
            label -- the fused labels
        """
        if ts is None:
            ts = np.arange( min(labels0.timestamp[0], labels1.timestamp[0]), max(labels0.timestamp[-1], labels1.timestamp[-1]), dt)
        fused_labels = np.zeros(ts.shape) + EyeClassifier.LABEL_NOISE
        for idx in range(ts.size):
            tprev = ts[idx-1] if idx > 0 else -np.inf
            tnext = ts[idx]
            cur_labels = np.concatenate( ( labels0.label[np.logical_and(labels0.timestamp >= tprev, labels0.timestamp < tnext)],
                                           labels1.label[np.logical_and(labels1.timestamp >= tprev, labels1.timestamp < tnext)]
                                            ) )
            ct_sac = np.count_nonzero(cur_labels == EyeClassifier.LABEL_SAC)
            ct_fix = np.count_nonzero(cur_labels == EyeClassifier.LABEL_FIX)
            
            if ct_fix > ct_sac:
                fused_labels[idx] = EyeClassifier.LABEL_FIX
            elif ct_sac > ct_fix or ct_sac > 0:# they're equal and it's not entirely noise
                fused_labels[idx] = EyeClassifier.LABEL_SAC
            elif cur_labels.size == 0 and idx > 0: # there's no data available
                # just keep the label
                fused_labels[idx] = fused_labels[idx-1]
            # otherwise leave as noise
        
        return pd.DataFrame({'timestamp': ts, 'label': fused_labels})
        
    
    def predict(self, data0, data1, ts=None, dt=None):
        labels0 = EyeClassifier._predict(self.bmm_eye0, self.eye0_labels, data0)
        labels1 = EyeClassifier._predict(self.bmm_eye1, self.eye1_labels, data1)
        labels0 = EyeClassifier.postprocess(labels0)
        labels1 = EyeClassifier.postprocess(labels1)
        return EyeClassifier.fuse(
                pd.DataFrame({'timestamp': data0.timestamp, 'label': labels0}),
                pd.DataFrame({'timestamp': data1.timestamp, 'label': labels1}),
                ts, dt
            )
        
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

        is_fix = (labels.label.values == EyeClassifier.LABEL_FIX).astype(np.int8)
        fix_change = is_fix[1:] - is_fix[:-1]

        fix_start = np.nonzero(fix_change == 1)[0] + 1
        if is_fix[0]:
            fix_start = np.concatenate( (0, fix_start) )
        fix_end = np.nonzero(fix_change == -1)[0]
        if is_fix[-1]:
            fix_end = np.concatenate( (fix_end, [is_fix.size-1]))

        fix = pd.DataFrame({ 'start_timestamp': labels.timestamp.values[fix_start], 'duration': (labels.timestamp.values[fix_end] - labels.timestamp.values[fix_start]) * 1000. })

        if min_fix_dur is not None:
            fix = fix.loc[fix.duration >= min_fix_dur, :]
        if gaze_data is not None:
            m_x = [ np.mean( gaze_data.x[np.logical_and(gaze_data.timestamp.values >= r.start_timestamp,
                                                                       gaze_data.timestamp.values <= r.start_timestamp + .001*r.duration)] )
                   for r in fix.itertuples() ]
            m_y = [ np.mean( gaze_data.y[np.logical_and(gaze_data.timestamp.values >= r.start_timestamp,
                                                                       gaze_data.timestamp.values <= r.start_timestamp + .001*r.duration)] )
                   for r in fix.itertuples() ]
            fix = fix.assign(x=m_x, y=m_y)
        return fix
    
    def get_fixations(self, data0, data1, ts=None, dt=None, gaze_data=None, min_fix_dur=100):
        if ts is None and dt is None and gaze_data is not None:
            ts = gaze_data.timestamp
        labels = self.predict(data0, data1, ts, dt)
        return EyeClassifier.get_fixations_from_labels(labels, gaze_data, min_fix_dur)
            
        
        
