from statsmodels.tsa.stattools import coint as ci
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pyEDM as edm
import pandas as pd
from math import isnan
from hyppo.time_series import MGCX, DcorrX
import warnings
from pynats.base import directed, undirected, parse_bivariate, unsigned, signed

import importlib
import scipy.spatial.distance as distance
import tslearn.metrics
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient, softdtw_barycenter

class coint(undirected,unsigned):
    
    humanname = "Cointegration"
    name = "coint"
    labels = ['unsigned','temporal','undirected','lagged']

    def __init__(self,method='johansen',statistic='trace_stat'):
        self._method = method
        self._statistic = statistic
        self.name = self.name + '_' + method + '_' + statistic

    # Return the negative t-statistic (proxy for how co-integrated they are)
    @parse_bivariate
    def bivariate(self,data,i=None,j=None,verbose=False):

        z = data.to_numpy(squeeze=True)
        M = data.n_processes

        if not hasattr(data,'coint'):
            data.coint = {'max_eig_stat': np.full((M, M), np.NaN), 'trace_stat': np.full((M, M), np.NaN),
                            'tstat': np.full((M, M), np.NaN), 'pvalue': np.full((M, M), np.NaN)}

        if self._method == 'johansen':
            if isnan(data.coint['max_eig_stat'][i,j]):
                z_ij_T = np.transpose(z[[i,j]])
                stats = coint_johansen(z_ij_T,det_order=0,k_ar_diff=1)
                data.coint['max_eig_stat'][i,j] = stats.max_eig_stat[0]
                data.coint['trace_stat'][i,j] = stats.trace_stat[0]
        elif self._method == 'aeg':
            if isnan(data.coint['tstat'][i,j]):
                stats = ci(z[i],z[j])
                data.coint['tstat'][i,j] = stats[0]
                data.coint['pvalue'][i,j] = stats[1]
        else:
            raise TypeError(f'Unknown statistic: {self._method}')

        return data.coint[self._statistic][i,j]

class ccm(directed,unsigned):

    humanname = "Convergent cross-maping"
    name = "ccm"
    labels = ['embedding','temporal','directed','lagged','causal']

    def __init__(self,statistic='mean'):
        self._statistic = statistic
        self.name = self.name + '_' + statistic
        if statistic == 'diff':
            self.issigned = lambda : True
            self.labels += ['signed']
        else:
            self.labels += ['unsigned']

    @staticmethod
    def _precompute(data):
        z = data.to_numpy(squeeze=True)

        M = data.n_processes
        N = data.n_observations
        df = pd.DataFrame(range(0,N),columns=['index'])
        embedding = np.zeros((M,1))

        names = []

        # First pass: infer optimal embedding
        for _i in range(M):
            names.append('var' + str(_i))
            df[names[_i]] = z[_i]
            pred = str(10) + ' ' + str(N-10)
            embed_df = edm.EmbedDimension(dataFrame=df,lib=pred,
                                            pred=pred,columns=str(_i),showPlot=False)
            embedding[_i] = embed_df.iloc[embed_df.idxmax().rho,0]
        
        # Get some reasonable library lengths
        nlibs = 5

        # Second pass: compute CCM
        score = np.zeros((M,M,nlibs+1))
        for _i in range(M):
            for _j in range(_i+1,M):
                E = int(max(embedding[[_i,_j]]))
                upperE = int(np.floor((N-E-1)/10)*10)
                lowerE = int(np.ceil(2*E/10)*10)
                inc = int((upperE-lowerE) / nlibs)
                lib_sizes = str(lowerE) + ' ' + str(upperE) + ' ' + str(inc)
                ccm_df = edm.CCM(dataFrame=df,E=E,columns=names[_i],target=names[_j],
                                    libSizes=lib_sizes,sample=100)
                sc1 = ccm_df.iloc[:,1]
                sc2 = ccm_df.iloc[:,2]
                score[_i,_j] = np.array(sc1)
                score[_j,_i] = np.array(sc2)

        data.ccm = {'embedding': embedding, 'score': score}

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        if not hasattr(data,'ccm'):
            ccm._precompute(data)

        if self._statistic == 'mean':
            stat = np.nanmean(data.ccm['score'][i,j])
        elif self._statistic == 'max':
            stat = np.nanmax(data.ccm['score'][i,j])
        elif self._statistic == 'diff':
            stat = np.nanmean(data.ccm['score'][i,j] - data.ccm['score'][j,i])
        else:
            raise TypeError(f'Unknown statistic: {self._statistic}')

        return stat

class dcorrx(directed,unsigned):
    """ Multi-graph correlation for time series
    """

    humanname = "Multi-scale graph correlation"
    name = "dcorrx"
    labels = ['unsigned','independence','temporal','directed','lagged']

    def __init__(self,max_lag=1):
        self._max_lag = max_lag
        self.name += f'_maxlag-{max_lag}'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _ = DcorrX(max_lag=self._max_lag).statistic(x,y)
        return stat

class mgcx(directed,unsigned):
    """ Multi-graph correlation for time series
    """

    humanname = "Multi-scale graph correlation"
    name = "mgcx"
    labels = ['unsigned','independence','temporal','directed','lagged']

    def __init__(self,max_lag=1):
        self._max_lag = max_lag
        self.name += f'_maxlag-{max_lag}'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _, _ = MGCX(max_lag=self._max_lag).statistic(x,y)
        return stat

class time_warping(undirected, unsigned):

    labels = ['unsigned','distance','temporal','undirected','lagged']

    def __init__(self,global_constraint='itakura'):
        self.name += '_' + global_constraint
        self._global_constraint = global_constraint

    @property
    def simfn(self):
        try:
            return self._simfn
        except AttributeError:
            raise NotImplementedError(f'Add the similarity function for {self.name}')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        return self._simfn(z[i],z[j],global_constraint=self._global_constraint)

class dynamic_time_warping(time_warping):

    humanname = 'Dynamic time warping'
    name = 'dtw'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.dtw

class canonical_time_warping(time_warping):

    humanname = 'Canonical time warping'
    name = 'ctw'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.ctw    

class longest_common_subsequence(time_warping):

    humanname = 'Longest common subsequence'
    name = 'lcss'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.lcss

class soft_dynamic_time_warping(time_warping):

    humanname = 'Dynamic time warping'
    name = 'softdtw'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.soft_dtw(z[i],z[j])

class global_alignment_kernel(time_warping):

    humanname = 'Global alignment kernel'
    name = 'gak'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.gak(z[i],z[j])

class lb_keogh(unsigned,directed):
    humanname = 'LB Keogh'
    name = 'lbk'
    labels = ['unsigned','distance','temporal','undirected','lagged']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.lb_keogh(ts_query=z[j],ts_candidate=z[j])

class barycenter(undirected,signed):

    humanname = 'Barycenter'
    name = 'bary'
    labels = ['signed','undirected','unpaired']

    def __init__(self,mode='euclidean',statistic='mean'):
        if mode == 'euclidean':
            self._fn = euclidean_barycenter
        elif mode == 'dtw':
            self._fn = dtw_barycenter_averaging
        elif mode == 'sgddtw':
            self._fn = dtw_barycenter_averaging_subgradient
        elif mode == 'softdtw':
            self._fn = softdtw_barycenter
        else:
            raise NameError(f'Unknown barycenter mode: {mode}')
        self._mode = mode

        if statistic == 'mean':
            self._statfn = np.nanmean
        elif statistic == 'max':
            self._statfn = np.nanmax
        else:
            raise NameError(f'Unknown statistic: {statistic}')

        self.name += f'_{mode}_{statistic}'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):

        try:
            barycenter = data.barycenter[self._mode]
        except (AttributeError,KeyError):
            z = data.to_numpy(squeeze=True)
            barycenter = self._fn(z)
            try:
                data.barycenter[self._mode] = barycenter
            except AttributeError:
                data.barycenter = {self._mode: barycenter}
        
        return self._statfn(barycenter)
