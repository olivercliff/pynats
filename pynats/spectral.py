import numpy as np
import spectral_connectivity as sc # For directed spectral measures (excl. spectral GC) 
from pynats.base import directed, parse_bivariate, undirected, parse_multivariate, unsigned
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu
from mne.connectivity import envelope_correlation as pec
import warnings

"""
The measures here come from three different toolkits:

    - Most measures come from Eden Kramer Lab's spectral_connectivity toolkit
    - Spectral Granger causality comes from nitime (since Kramer's version doesn't optimise AR order, e.g., using BIC).

    - [Outdated] Simple undirected measurements generally come from MNE (coherence, imaginary coherence, phase slope index)
        - I originally used this b/c they had additional (short-time fourier and Mortlet) ways of computing the spectral measures, but Kramer has more measures. May be worth putting both in and ensuring they're identical.

Granger causality could be computed from the VAR models in the infotheory module but this involves pretty intense integration with the temporal toolkits so may not ever get done.
"""

class kramer(unsigned):

    def __init__(self,fs=1,fmin=0,fmax=None,statistic='mean'):
        if fmax is None:
            fmax = fs/2
            
        self._fs = fs
        if fs != 1:
            warnings.warn('Multiple sampling frequencies not yet handled.')
        self._fmin = fmin
        self._fmax = fmax
        if statistic == 'mean':
            self._statfn = np.nanmean
        elif statistic == 'max':
            self._statfn = np.nanmax
        else:
            raise NameError(f'Unknown statistic: {statistic}')
        self._statistic = statistic
        paramstr = f'_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
        self.name = self.name + paramstr

    @property
    def measure(self):
        try:
            return self._measure
        except AttributeError:
            raise AttributeError(f'Include measure for {self.humanname}')

    def _get_measure(self,C):
        raise NotImplementedError

class kramer_mv(kramer):

    def _get_cache(self,data):
        try:
            res = data.kramer_mv[self.measure]
            freq = data.kramer_mv['freq']
        except (AttributeError,KeyError):
            z = np.transpose(data.to_numpy(squeeze=True))
            m = sc.Multitaper(z,sampling_frequency=self._fs)
            conn = sc.Connectivity.from_multitaper(m)
            try:
                res = getattr(conn,self.measure)()
            except TypeError:
                res = self._get_measure(conn)

            freq = conn.frequencies
            try:
                data.kramer_mv[self.measure] = res
            except AttributeError:
                data.kramer_mv = {'freq': freq, self.measure: res}

        return res, freq

    @parse_multivariate
    def adjacency(self, data):
        adj_freq, freq = self._get_cache(data)
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        try:
            adj = self._statfn(np.real(adj_freq[0,freq_id,:,:]), axis=0)
        except IndexError: # For phase-slope index
            adj = adj_freq[0]
        except TypeError: # For group delay
            adj = adj_freq[1][0]
        np.fill_diagonal(adj,np.nan)
        return adj

class kramer_bv(kramer):

    def _get_cache(self,data,i,j):
        key = (self.measure,i,j)
        try:
            res = data.kramer_bv[key]
            freq = data.kramer_bv['freq']
        except (KeyError,AttributeError):
            z = np.transpose(data.to_numpy(squeeze=True)[[i,j]])
            m = sc.Multitaper(z,sampling_frequency=self._fs)
            conn = sc.Connectivity.from_multitaper(m)
            try:
                res = getattr(conn,self.measure)()
            except TypeError:
                res = self._get_measure(conn)

            freq = conn.frequencies
            try:
                data.kramer_bv[key] = res
            except AttributeError:
                data.kramer_bv = {'freq': freq, key: res}

        return res, freq

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """ TODO: cache this result
        """
        bv_freq, freq = self._get_cache(data,i,j)
        freq_id = np.where((freq > self._fmin) * (freq < self._fmax))[0]

        return self._statfn(bv_freq[0,freq_id,0,1])

class coherency(kramer_mv,undirected):
    humanname = 'Coherency'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'coh'
        super().__init__(**kwargs)
        self._measure = 'coherency'

class coherence_phase(kramer_mv,undirected):
    humanname = 'Coherence phase'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'phase'
        super().__init__(**kwargs)
        self._measure = 'coherence_phase'

class coherence_magnitude(kramer_mv,undirected):
    humanname = 'Coherence magnitude'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'cohmag'
        super().__init__(**kwargs)
        self._measure = 'coherence_magnitude'

class icoherence(kramer_mv,undirected):
    humanname = 'Imaginary coherence'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'icoh'
        super().__init__(**kwargs)
        self._measure = 'imaginary_coherence'

class phase_locking_value(kramer_mv,undirected):
    humanname = 'Phase-locking value'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'plv'
        super().__init__(**kwargs)
        self._measure = 'phase_locking_value'

class phase_lag_index(kramer_mv,undirected):
    humanname = 'Phase-lag index'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'pli'
        super().__init__(**kwargs)
        self._measure = 'phase_lag_index'

class weighted_phase_lag_index(kramer_mv,undirected):
    humanname = 'Weighted phase-lag index'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'wpli'
        super().__init__(**kwargs)
        self._measure = 'weighted_phase_lag_index'

class debiased_squared_phase_lag_index(kramer_mv,undirected):
    humanname = 'Debiased squared phase-lag index'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'dspli'
        super().__init__(**kwargs)
        self._measure = 'debiased_squared_phase_lag_index'

class debiased_squared_weighted_phase_lag_index(kramer_mv,undirected):
    humanname = 'Debiased squared weighted phase-lag index'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'dswpli'
        super().__init__(**kwargs)
        self._measure = 'debiased_squared_weighted_phase_lag_index'

class pairwise_phase_consistency(kramer_mv,undirected):
    humanname = 'Pairwise phase consistency'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'ppc'
        super().__init__(**kwargs)
        self._measure = 'pairwise_phase_consistency'

"""
    These next several seem to segfault for large vector autoregressive processes (something to do with np.linalg solver).
    Switched them to bivariate for now until the issue is resolved
"""
class directed_coherence(kramer_bv,directed):
    humanname = 'Directed coherence'
    labels = ['unsigned','spectral','directed']

    def __init__(self,**kwargs):
        self.name = 'dcoh'
        super().__init__(**kwargs)
        self._measure = 'directed_coherence'

class partial_directed_coherence(kramer_bv,directed):
    humanname = 'Partial directed coherence'
    labels = ['unsigned','spectral','directed']

    def __init__(self,**kwargs):
        self.name = 'pdcoh'
        super().__init__(**kwargs)
        self._measure = 'partial_directed_coherence'

class generalized_partial_directed_coherence(kramer_bv,directed):
    humanname = 'Generalized partial directed coherence'
    labels = ['unsigned','spectral','directed']

    def __init__(self,**kwargs):
        self.name = 'gpdcoh'
        super().__init__(**kwargs)
        self._measure = 'generalized_partial_directed_coherence'

class directed_transfer_function(kramer_bv,directed):
    humanname = 'Directed transfer function'
    labels = ['unsigned','spectral','directed','lagged']

    def __init__(self,**kwargs):
        self.name = 'dtf'
        super().__init__(**kwargs)
        self._measure = 'directed_transfer_function'

class direct_directed_transfer_function(kramer_bv,directed):
    humanname = 'Direct directed transfer function'
    labels = ['unsigned','spectral','directed','lagged']

    def __init__(self,**kwargs):
        self.name = 'ddtf'
        super().__init__(**kwargs)
        self._measure = 'direct_directed_transfer_function'

class phase_slope_index(kramer_mv,directed):
    humanname = 'Phase slope index'
    labels = ['unsigned','spectral','directed']

    def __init__(self,**kwargs):
        self.name = 'psi'
        super().__init__(**kwargs)
        self._measure = 'phase_slope_index'
    
    def _get_measure(self,C):
        return C.phase_slope_index(frequencies_of_interest=[self._fmin,self._fmax],
                                    frequency_resolution=(self._fmax-self._fmin)/50)

class group_delay(kramer_mv,directed):
    humanname = 'Group delay'
    labels = ['unsigned','spectral','directed','lagged']

    def __init__(self,**kwargs):
        self.name = 'gd'
        super().__init__(**kwargs)
        self._measure = 'group_delay'
    
    def _get_measure(self,C):
        return C.group_delay(frequencies_of_interest=[self._fmin,self._fmax],
                            frequency_resolution=(self._fmax-self._fmin)/50)

class partial_coherence(undirected,unsigned):
    humanname = 'Partial coherence'
    name = 'pcoh'
    labels = ['unsigned','spectral','directed']

    def __init__(self,fs=1,fmin=0.05,fmax=np.pi/2,statistic='mean'):
        self._TR = 1/fs # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        if statistic == 'mean':
            self._statfn = np.mean
        elif statistic == 'max':
            self._statfn = np.max
        else:
            raise NameError(f'Unknown statistic {statistic}')
        paramstr = f'_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
        self.name = self.name + paramstr

    @parse_multivariate
    def adjacency(self,data):        
        # This should be changed to conditioning on all, rather than averaging all conditionals

        if not hasattr(data,'pcoh'):
            z = np.squeeze(data.to_numpy())
            pdata = tsu.percent_change(z)
            time_series = ts.TimeSeries(pdata, sampling_interval=1)
            C1 = nta.CoherenceAnalyzer(time_series)
            data.pcoh = {'gamma': np.nanmean(C1.coherence_partial,axis=2), 'freq': C1.frequencies}

        freq_idx_C = np.where((data.pcoh['freq'] > self._fmin) * (data.pcoh['freq'] < self._fmax))[0]
        pcoh = self._statfn(data.pcoh['gamma'][:, :, freq_idx_C], -1)
        np.fill_diagonal(pcoh,np.nan)
        return pcoh

class spectral_granger(directed,unsigned):
    humanname = 'Spectral Granger causality'
    name = 'sgc'
    labels = ['unsigned','embedding','spectral','directed','lagged']

    def __init__(self,fs=1,fmin=0.0,fmax=0.5,order=None,max_order=50,statistic='mean'):
        self._fs = fs # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        self._order = order
        self._max_order = max_order
        if statistic == 'mean':
            self._statfn = np.mean
        elif statistic == 'max':
            self._statfn = np.max
        else:
            raise NameError(f'Unknown statistic {statistic}')

        paramstr = f'_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}_order-{order}'.replace('.','-')
        self.name = self.name + paramstr

    def _getkey(self):
        return (self._order,self._max_order)

    def _get_cache(self,data):
        key = self._getkey()
        try:
            F = data.spectral_granger[key]
        except (AttributeError,KeyError):
            z = data.to_numpy(squeeze=True)
            time_series = ts.TimeSeries(z,sampling_interval=1)
            F = nta.GrangerAnalyzer(time_series, order=self._order, max_order=self._max_order)
            try:
                data.spectral_granger[key] = F
            except AttributeError:
                data.spectral_granger = {key: F}
        return F

    @parse_multivariate
    def adjacency(self,data):
        F = self._get_cache(data)
        m = data.n_processes

        freq_id = np.where((F.frequencies >= self._fmin) * (F.frequencies <= self._fmax))[0]
    
        gc_triu = self._statfn(F.causality_xy[...,freq_id], axis=-1)
        gc_tril = self._statfn(F.causality_yx[...,freq_id], axis=-1)

        gc = np.empty((m,m))
        triu_id = np.triu_indices(m)

        gc[triu_id] = gc_triu[triu_id]
        gc[triu_id[1],triu_id[0]] = gc_tril[triu_id]

        return gc

class envelope_correlation(undirected):
    humanname = 'Power envelope correlation'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,orth=False,log=False,absolute=False):
        self.name = 'pec'
        self._orth = False
        if orth:
            self._orth = 'pairwise'
            self.name += '_orth'
        self._log = log
        if log:
            self.name += '_log'
        self._absolute = absolute
        if absolute:
            self.name += '_abs'

    @parse_multivariate
    def adjacency(self, data):
        z = np.moveaxis(data.to_numpy(),2,0)
        adj = np.squeeze(pec(z,orthogonalize=self._orth,log=self._log,absolute=self._absolute))
        np.fill_diagonal(adj,np.nan)
        return adj