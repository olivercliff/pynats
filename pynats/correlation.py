from nilearn import connectome as fc
from sklearn import covariance as cov
from scipy import stats, spatial, signal
import numpy as np
from pynats.base import undirected, directed, parse_bivariate, parse_multivariate, signed, unsigned
from hyppo.independence import MGC, Dcorr, HHG, Hsic
import warnings

class connectivity(undirected,signed):
    """ Base class for (functional) connectivity-based measures 
    
    Information on covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html
    """

    humanname = "Pearon's product-moment correlation coefficient"
    labels = ['correlation','connectivity','unordered','linear','undirected']

    _ledoit_wolf = cov.LedoitWolf
    _empirical = cov.EmpiricalCovariance
    _shrunk = cov.ShrunkCovariance
    _oas = cov.OAS

    def __init__(self, kind, squared=False,
                 cov_estimator='empirical'):
        paramstr = f'_{cov_estimator}'
        if squared:
            paramstr = '-sq' + paramstr
            self.labels += ['unsigned']
            self.issigned = lambda : False
        else:
            self.labels += ['signed']
        self.name = self.name + paramstr
        self._squared = squared
        self._cov_estimator = eval('self._' + cov_estimator + '()')
        self._kind = kind

    @parse_multivariate
    def adjacency(self,data):
        z = np.swapaxes(data.to_numpy(),2,0)
        fc_measure = fc.ConnectivityMeasure(cov_estimator=self._cov_estimator,
                                                     kind=self._kind)

        fc_matrix = fc_measure.fit_transform(z)[0]
        np.fill_diagonal(fc_matrix,np.nan)
        if self._squared:
            return np.square(fc_matrix)
        else:
            return fc_matrix

class pearsonr(connectivity):

    humanname = "Pearson's product-moment correlation"

    def __init__(self,cov_estimator='empirical',squared=False):
        self.name = 'pearsonr'
        super(pearsonr,self).__init__(kind='correlation',squared=squared,cov_estimator=cov_estimator)

class pcor(connectivity):

    humanname = "Partial correlation"

    def __init__(self,cov_estimator='empirical',squared=False):
        self.name = 'pcorr'
        super(pcor,self).__init__(kind='partial correlation',squared=squared,cov_estimator=cov_estimator)

class tangent(connectivity):

    humanname = "Tangent"

    def __init__(self,cov_estimator='empirical',squared=False):
        self.name = 'tangent'
        super(tangent,self).__init__('tangent',squared=squared,cov_estimator=cov_estimator)

class covariance(connectivity):

    humanname = "Covariance"
    name = "cov"

    def __init__(self,cov_estimator='empirical',squared=False):
        self.name = 'cov'
        super(covariance,self).__init__('covariance',squared=squared,cov_estimator=cov_estimator)

class precision(connectivity):

    humanname = "Precision"

    def __init__(self,cov_estimator='empirical',squared=False):
        self.name = 'prec'
        super(precision,self).__init__('precision',squared=squared,cov_estimator=cov_estimator)

class xcorr(undirected,signed):

    humanname = "Cross correlation"
    labels = ['correlation','unordered','lagged','linear','undirected']

    def __init__(self,squared=False,statistic='max'):
        self.name = 'xcorr'
        self._squared = squared
        self._statistic = statistic

        if self._squared:
            self.issigned = lambda : False
            self.name = self.name + '-sq'
            self.labels += ['unsigned']
        else:
            self.labels += ['signed']
        self.name = self.name + '_' + statistic
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None):

        if not hasattr(data,'xcorr'):
            data.xcorr = np.ones((data.n_processes,data.n_processes,data.n_observations*2-1)) * -np.inf

        if data.xcorr[i,j,0] == -np.inf:
            x, y = data.to_numpy()[[i,j]]
            data.xcorr[i,j] = np.squeeze(signal.correlate(x,y,'full'))
            data.xcorr[i,j] = data.xcorr[i,j] / x.std() / y.std() / (data.n_observations - 1)

        if self._statistic == 'max':
            if self._squared:
                return np.max(data.xcorr[i,j]**2)
            else:
                return np.max(data.xcorr[i,j])
        elif self._statistic == 'mean':
            if self._squared:
                return np.mean(data.xcorr[i,j]**2)
            else:
                return np.mean(data.xcorr[i,j])
        else:
            raise TypeError(f'Unknown statistic: {self._statistic}') 

class spearmanr(undirected,signed):

    humanname = "Spearman's correlation coefficient"
    name = "spearmanr"
    labels = ['correlation','unordered','rank','linear','undirected']

    def __init__(self,squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda : False
            self.name = self.name + '-sq'
            self.labels += ['unsigned']
        else:
            self.labels += ['signed']
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        if self._squared:
            return stats.spearmanr(x,y).correlation ** 2
        else:
            return stats.spearmanr(x,y).correlation

class kendalltau(undirected,signed):

    humanname = "Kendall's tau"
    name = "kendalltau"
    labels = ['correlation','unordered','rank','linear','undirected']

    def __init__(self,squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda : False
            self.name = self.name + '-sq'
            self.labels += ['unsigned']
        else:
            self.labels += ['signed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        if self._squared:
            return stats.kendalltau(x,y).correlation ** 2
        else:
            return stats.kendalltau(x,y).correlation

""" TODO: include optional kernels in each method
"""
class hsic(undirected,unsigned):
    """ Hilbert-Schmidt Independence Criterion (Hsic)
    """

    humanname = "Hilbert-Schmidt Independence Criterion"
    name = 'hsic'
    labels = ['independence','unordered','nonlinear','undirected']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        stat, _ = Hsic().test(x, y, auto=True )
        return stat

class hhg(directed,unsigned):
    """ Heller-Heller-Gorfine independence criterion
    """

    humanname = "Heller-Heller-Gorfine Independence Criterion"
    name = 'hhg'
    labels = ['independence','unordered','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _ = HHG().test(x, y, reps=0)
        return stat

class dcorr(undirected,unsigned):
    """ Correlation of distances
    """

    humanname = "Distance correlation"
    name = 'dcorr'
    labels = ['independence','unordered','nonlinear','undirected']
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """
        """
        x, y = data.to_numpy()[[i,j]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _ = Dcorr().test(x, y, auto=True, reps=0)
        return stat

class mgc(undirected,unsigned):
    """ Multi-graph correlation
    """

    humanname = "Multi-scale graph correlation"
    name = "mgc"
    labels = ['independence','unordered','nonlinear','undirected']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _, _ = MGC().test(x, y, reps=0 )
        return stat