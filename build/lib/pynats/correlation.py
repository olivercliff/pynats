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

    def __init__(self,squared=False,statistic='max',sigonly=True):
        self.name = 'xcorr'
        self._squared = squared
        self._statistic = statistic
        self._sigonly = sigonly

        if self._squared:
            self.issigned = lambda : False
            self.name = self.name + '-sq'
            self.labels += ['unsigned']
        else:
            self.labels += ['signed']
        self.name += f'_{statistic}_sig-{sigonly}'
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None):

        T = data.n_observations
        try: 
            r_ij = data.xcorr[(i,j)]
        except (KeyError,AttributeError):
            x, y = data.to_numpy()[[i,j]]

            r_ij = np.squeeze(signal.correlate(x,y,'full'))
            r_ij = r_ij / x.std() / y.std() / (T-1)

            # Truncate to T/4
            r_ij = r_ij[T-T//4:T+T//4]

            try:
                data.xcorr[(i,j)] = r_ij
            except AttributeError:
                data.xcorr = {(i,j): r_ij}

        # Truncate at first statistically significant zero (i.e., |r| <= 1.96/sqrt(T))
        if self._sigonly:
            N = len(r_ij)//2
            fzf = np.where(np.abs(r_ij[len(r_ij)//2:]) <= 1.96/np.sqrt(N))[0][0]
            fzr = np.where(np.abs(r_ij[:len(r_ij)//2]) <= 1.96/np.sqrt(N))[0][-1]
            r_ij = r_ij[N-fzr:N+fzf]

        if self._statistic == 'max':
            if self._squared:
                return np.max(r_ij**2)
            else:
                return np.max(r_ij)
        elif self._statistic == 'mean':
            if self._squared:
                return np.mean(r_ij**2)
            else:
                return np.mean(r_ij)
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

    def __init__(self,biased=False):
        self._biased = biased
        if biased:
            self.name += '_biased'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        stat = Hsic(bias=self._biased).statistic(x,y)
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
            stat = HHG().statistic(x,y)
        return stat

class dcorr(undirected,unsigned):
    """ Distance correlation
    """

    humanname = "Distance correlation"
    name = 'dcorr'
    labels = ['independence','unordered','nonlinear','undirected']

    def __init__(self,biased=False):
        self._biased = biased
        if biased:
            self.name += '_biased'
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """
        """
        x, y = data.to_numpy()[[i,j]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat = Dcorr(bias=self._biased).statistic(x,y)
        return stat

class mgc(undirected,unsigned):
    """ Multiscale graph correlation
    """

    humanname = "Multiscale graph correlation"
    name = "mgc"
    labels = ['independence','unordered','nonlinear','undirected']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat = MGC().statistic(x,y)
        return stat