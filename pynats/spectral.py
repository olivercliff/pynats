import numpy as np
import spectral_connectivity as sc # For directed spectral measures (excl. spectral GC) 
from pynats.base import directed, parse_bivariate, undirected, parse_multivariate, unsigned
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu
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

    def __init__(self,fs=1,fmin=0,fmax=None):
        if fmax is None:
            fmax = fs/2
            
        self._fs = fs
        if fs != 1:
            warnings.warning('Multiple sampling frequencies not yet handled.')
        self._fmin = fmin
        self._fmax = fmax
        paramstr = f'_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
        self.name = self.name + paramstr

    def _get_measure(self,C):
        raise NotImplementedError

class kramer_mv(kramer):

    def _get_cache(self,data):
        try:
            conn = data.kramer_mv
        except AttributeError:
            z = np.transpose(data.to_numpy(squeeze=True))
            m = sc.Multitaper(z,sampling_frequency=self._fs)
            conn = data.kramer_mv = sc.Connectivity.from_multitaper(m)

        freq = conn.frequencies
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        return conn, freq_id

    @parse_multivariate
    def adjacency(self, data):
        conn, freq_id = self._get_cache(data)
        
        adj_freq = self._get_measure(conn)
        try:
            adj = np.nanmean(np.real(adj_freq[0,freq_id,:,:]), axis=0)
        except IndexError: # For phase-slope index
            adj = adj_freq[0]
        except TypeError: # For group delay
            adj = adj_freq[1][0]
        np.fill_diagonal(adj,np.nan)
        return adj

class kramer_bv(kramer):

    def _get_cache(self,data,i,j):
        try:
            conn = data.kramer_bv[(i,j)]
        except (KeyError,AttributeError) as err:
            z = np.transpose(data.to_numpy(squeeze=True)[[i,j]])
            m = sc.Multitaper(z,sampling_frequency=self._fs)
            conn = sc.Connectivity.from_multitaper(m)
            if isinstance(err,AttributeError):
                data.kramer_bv = {(i,j): conn}
            else:
                data.kramer_bv[(i,j)] = conn

        freq = conn.frequencies
        freq_id = np.where((freq > self._fmin) * (freq < self._fmax))[0]

        return conn, freq_id

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """ TODO: cache this result
        """
        conn, freq_id = self._get_cache(data,i,j)
        bv_freq = self._get_measure(conn)
        return np.nanmean(bv_freq[0,freq_id,0,1])

class coherency(kramer_mv,undirected):
    humanname = 'Coherency'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'coh'
        super().__init__(**kwargs)

    def _get_measure(self,C):
        return C.coherency()

class coherence_phase(kramer_mv,undirected):
    humanname = 'Coherence phase'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'phase'
        super().__init__(**kwargs)

    def _get_measure(self,C):
        return C.coherence_phase()

class coherence_magnitude(kramer_mv,undirected):
    humanname = 'Coherence magnitude'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'cohmag'
        super().__init__(**kwargs)

    def _get_measure(self,C):
        return C.coherence_magnitude()

class icoherence(kramer_mv,undirected):
    humanname = 'Coherence'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'icoh'
        super().__init__(**kwargs)
        self._measure = 'imaginary_coherence'

    def _get_measure(self,C):
        return C.imaginary_coherence()

class phase_locking_value(kramer_mv,undirected):
    humanname = 'Phase-locking value'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'plv'
        super().__init__(**kwargs)
        self._measure = 'phase_locking_value'

    def _get_measure(self,C):
        return C.phase_locking_value()

class phase_lag_index(kramer_mv,undirected):
    humanname = 'Phase-lag index'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'pli'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.phase_lag_index()

class weighted_phase_lag_index(kramer_mv,undirected):
    humanname = 'Weighted phase-lag index'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'wpli'
        super().__init__(**kwargs)
        
    def _get_measure(self,C):
        return C.weighted_phase_lag_index()

class debiased_squared_phase_lag_index(kramer_mv,undirected):
    humanname = 'Debiased squared phase-lag index'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'dspli'
        super().__init__(**kwargs)
        
    def _get_measure(self,C):
        return C.debiased_squared_phase_lag_index()

class debiased_squared_weighted_phase_lag_index(kramer_mv,undirected):
    humanname = 'Debiased squared weighted phase-lag index'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'dswpli'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.debiased_squared_weighted_phase_lag_index()

class pairwise_phase_consistency(kramer_mv,undirected):
    humanname = 'Pairwise phase consistency'
    labels = ['unsigned','spectral','undirected']

    def __init__(self,**kwargs):
        self.name = 'ppc'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.pairwise_phase_consistency()

class directed_coherence(kramer_mv,directed):
    humanname = 'Directed coherence'
    labels = ['unsigned','spectral','directed']

    def __init__(self,**kwargs):
        self.name = 'dcoh'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.directed_coherence()

class partial_directed_coherence(kramer_mv,directed):
    humanname = 'Partial directed coherence'
    labels = ['unsigned','spectral','directed']

    def __init__(self,**kwargs):
        self.name = 'pdcoh'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.partial_directed_coherence()

class generalized_partial_directed_coherence(kramer_mv,directed):
    humanname = 'Generalized partial directed coherence'
    labels = ['unsigned','spectral','directed']

    def __init__(self,**kwargs):
        self.name = 'gpdcoh'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.generalized_partial_directed_coherence()

"""
    These two seem to segfault for large vector autoregressive processes (something to do with np.linalg solver).
    Switched them to bivariate for now until the issue is resolved
"""
class directed_transfer_function(kramer_bv,directed):
    humanname = 'Directed transfer function'
    labels = ['unsigned','spectral','directed','lagged']

    def __init__(self,**kwargs):
        self.name = 'dtf'
        super().__init__(**kwargs)

    def _get_measure(self,C):
        return C.directed_transfer_function()

class direct_directed_transfer_function(kramer_bv,directed):
    humanname = 'Direct directed transfer function'
    labels = ['unsigned','spectral','directed','lagged']

    def __init__(self,**kwargs):
        self.name = 'ddtf'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.direct_directed_transfer_function()

class phase_slope_index(kramer_mv,directed):
    humanname = 'Phase slope index'
    labels = ['unsigned','spectral','directed']

    def __init__(self,**kwargs):
        self.name = 'psi'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.phase_slope_index(frequencies_of_interest=[self._fmin,self._fmax],
                                    frequency_resolution=0.1)

class group_delay(kramer_mv,directed):
    humanname = 'Group delay'
    labels = ['unsigned','spectral','directed','lagged']

    def __init__(self,**kwargs):
        self.name = 'gd'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.group_delay(frequencies_of_interest=[self._fmin,self._fmax],
                            frequency_resolution=0.1)

class partial_coherence(undirected,unsigned):
    humanname = 'Partial coherence'
    name = 'pcoh'
    labels = ['unsigned','spectral','directed']

    def __init__(self,fs=1,fmin=0.05,fmax=np.pi/2):
        self._TR = 1/fs # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        paramstr = f'_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
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
        pcoh = np.nan_to_num(np.mean(data.pcoh['gamma'][:, :, freq_idx_C], -1))
        np.fill_diagonal(pcoh,np.nan)
        return pcoh

class spectral_granger(directed,unsigned):
    humanname = 'Spectral Granger causality'
    name = 'sgc'
    labels = ['unsigned','embedding','spectral','directed','lagged']

    def __init__(self,fs=1,fmin=0.0,fmax=0.5,order=None,max_order=50):
        self._fs = fs # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        self._order = order
        self._max_order = max_order
        paramstr = f'_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}_order-{order}'.replace('.','-')
        self.name = self.name + paramstr

    @parse_multivariate
    def adjacency(self, data):
        z = data.to_numpy(squeeze=True)
        m = data.n_processes
 
        pdata = tsu.percent_change(z)
        time_series = ts.TimeSeries(pdata, sampling_interval=1)
        G = nta.GrangerAnalyzer(time_series, order=self._order, max_order=self._max_order)
        try:
            freq_idx_G = np.where((G.frequencies >= self._fmin) * (G.frequencies <= self._fmax))[0]
        
            gc_triu = np.mean(G.causality_xy[:,:,freq_idx_G], -1)
            gc_tril = np.mean(G.causality_yx[:,:,freq_idx_G], -1)

            gc = np.empty((m,m))
            triu_id = np.triu_indices(m)

            gc[triu_id] = gc_triu[triu_id]
            gc[triu_id[1],triu_id[0]] = gc_tril[triu_id]
        except (ValueError,TypeError) as err:
            print(f'Spectral GC failed: {err}')
            gc = np.empty((m,m))
            gc[:] = np.NaN

        return gc