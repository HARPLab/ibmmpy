#!/usr/bin/env python

import scipy.spatial.distance
import sklearn.mixture
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
            pos = np.hstack((data.loc[:, ['x','y']].values, np.ones(len(data), 1)))
        elif dist_method == DIST_METHOD_EUC:
            pos = data.loc[:, ['x','y']]
        else:
            raise RuntimeError("unreachable")
        
        # TODO: smoothing
        
        # Remove low-confidence values 
        pos[data['confidence'] < conf_thresh] = np.nan
        
        # Compute velocity
        if dist_method == DIST_METHOD_VECTOR:
            dist = np.arccos(scipy.spatial.distance.cdist(pos[1:,:], pos[:-1,:], 'cosine'))
        elif dist_method == DIST_METHOD_EUC:
            dist = scipy.spatial.distance.cdist(pos[1:,:], pos[:-1,:], 'euclidean')
        else:
            raise RuntimeError("unreachable")
        dt = np.diff(data['timestamp'].values)
        # add a nan value at the beginning so the data point count remains the same
        veloc = np.vstack((np.nan, dist / dt))
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
            self.eye0_labels = [EyeClassifier.LABEL_FIX, EyeClassifier.LABEL_SAC]
        else:
            self.eye0_labels = [EyeClassifier.LABEL_SAC, EyeClassifier.LABEL_FIX]
        EyeClassifier._fit(self.bmm_eye1, data1)
        if self.bmm_eye1.means_[0] < self.bmm_eye1.means_[1]:
            self.eye1_labels = [EyeClassifier.LABEL_FIX, EyeClassifier.LABEL_SAC]
        else:
            self.eye1_labels = [EyeClassifier.LABEL_SAC, EyeClassifier.LABEL_FIX]
    
    @staticmethod
    def _predict(model, model_labels, data):
        labels = np.ones(len(data), 1)*EyeClassifier.LABEL_NOISE
        valid_mask = np.logical_not(np.isnan(data['velocity']))
        labels[valid_mask] = model_labels[model.predict(data[valid_mask, 'velocity'])]
        return labels
    
    @staticmethod
    def postprocess(labels):
    
    def predict(self, data0, data1):
        labels0 = EyeClassifier._predict(self.bmm_eye0, self.eye0_labels, data0)
        labels1 = EyeClassifier._predict(self.bmm_eye1, self.eye1_labels, data1)
        

