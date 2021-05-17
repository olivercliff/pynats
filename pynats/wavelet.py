from mne.connectivity import spectral_connectivity
from pynats.base import directed, parse_bivariate, undirected, parse_multivariate, unsigned
import numpy as np
import warnings

class mne(unsigned):

    _measure_list = []

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

        # Probably not good practice
        if self._measure not in mne._measure_list:
            mne._measure_list.append(self._measure)

    def _get_measure(self,C):
        raise NotImplementedError

    def _get_cache(self,data):
        try:
            conn = data.mne['conn']
            freq = data.mne['freq']
        except AttributeError:
            z = np.moveaxis(data.to_numpy(),2,0)

            freqs = np.linspace(0.2, 0.5, 10)
            conn, freq, _, _, _ = spectral_connectivity(
                    data=z, method=mne._measure_list, mode='cwt_morlet',
                    sfreq=self._fs, mt_adaptive=True, cwt_freqs=freqs,
                    verbose='WARNING')
            data.mne = dict(conn=conn,freq=freq)

        # freq = conn.frequencies
        myconn = conn[[i for i, m in enumerate(self._measure_list) if m == self._measure][0]]
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        return myconn, freq_id

    @parse_multivariate
    def adjacency(self, data):
        adj_freq, freq_id = self._get_cache(data)
        adj = np.nanmean(np.real(adj_freq[:,:,freq_id]), axis=(2,3))
        np.fill_diagonal(adj,np.nan)
        return adj

class coherency(mne,undirected):
    humanname = 'Coherency (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'cwt_coh'
        self._measure = 'coh'
        super().__init__(**kwargs)

class icoherence(mne,undirected):
    humanname = 'Imaginary coherency (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'cwt_icoh'
        self._measure = 'imcoh'
        super().__init__(**kwargs)

class phase_locking_value(mne,undirected):
    humanname = 'Phase-locking value (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'cwt_plv'
        self._measure = 'plv'
        super().__init__(**kwargs)

class pairwise_phase_consistency(mne,undirected):
    humanname = 'Pairwise phase consistency (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'cwt_ppc'
        self._measure = 'ppc'
        super().__init__(**kwargs)

class phase_lag_index(mne,undirected):
    humanname = 'Phase lag index (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'cwt_pli'
        self._measure = 'pli'
        super().__init__(**kwargs)

class debiased_squared_phase_lag_index(mne,undirected):
    humanname = 'Debiased squared phase lag index (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'cwt_dspli'
        self._measure = 'pli2_unbiased'
        super().__init__(**kwargs)

class weighted_phase_lag_index(mne,undirected):
    humanname = 'Weighted squared phase lag index (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'cwt_wspli'
        self._measure = 'wpli'
        super().__init__(**kwargs)

class debiased_weighted_squared_phase_lag_index(mne,undirected):
    humanname = 'Debiased weighted squared phase lag index (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'cwt_dwspli'
        self._measure = 'wpli2_debiased'
        super().__init__(**kwargs)