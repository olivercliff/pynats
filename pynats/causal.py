from cdt.causality.pairwise import ANM, BivariateFit, CDS, GNN, IGCI, RECI
from pynats.base import directed, undirected, parse_bivariate, unsigned

class anm(directed,unsigned):

    humanname = "Additive noise model"
    name = 'anm'
    labels = ['unsigned','model based','causal','unordered','linear','directed']

    def __init__(self,statistic='score'):
        if statistic == 'score' or statistic == 'dir':
            self._statistic = statistic
            self.name += f'_{statistic}'
        else:
            raise NameError(f'Unknown statistic: {statistic}')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        if self._statistic == 'score':
            return ANM().anm_score(z[i], z[j])
        else:
            return ANM().predict_proba((z[i], z[j]))

class gpfit(directed,unsigned):
    
    humanname = 'Gaussian process bivariate fit'
    name = 'gpfit'
    labels = ['unsigned','model based','causal','unordered','normal','nonlinear','directed']

    def __init__(self,statistic='score'):
        if statistic == 'score' or statistic == 'dir':
            self._statistic = statistic
            self.name += f'_{statistic}'
        else:
            raise NameError(f'Unknown statistic: {statistic}')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()

        if self._statistic == 'score':
            return BivariateFit().b_fit_score(z[i], z[j])
        else:
            return BivariateFit().predict_proba((z[i], z[j]))

class cds(directed,unsigned):
    
    humanname = 'Conditional distribution similarity statistic'
    name = 'cds'
    labels = ['unsigned','model based','causal','unordered','nonlinear','directed']

    def __init__(self,statistic='score'):
        if statistic == 'score' or statistic == 'dir':
            self._statistic = statistic
            self.name += f'_{statistic}'
        else:
            raise NameError(f'Unknown statistic: {statistic}')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()

        if self._statistic == 'score':
            return CDS().cds_score(z[i], z[j])
        else:
            return CDS().predict_proba((z[i], z[j]))


class reci(directed,unsigned):

    humanname = 'Regression error-based causal inference'
    name = 'reci'
    labels = ['unsigned','causal','unordered','neural network','nonlinear','directed']

    def __init__(self,statistic='score'):
        if statistic == 'score' or statistic == 'dir':
            self._statistic = statistic
            self.name += f'_{statistic}'
        else:
            raise NameError(f'Unknown statistic: {statistic}')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()

        if self._statistic == 'score':
            return RECI().b_fit_score(z[i], z[j])
        else:
            return RECI().predict_proba((z[i], z[j]))

class gnn(undirected,unsigned):

    humanname = 'Shallow generative neural network'
    name = 'gnn'
    labels = ['unsigned','causal','unordered','neural network','nonlinear','undirected']

    def __init__(self):
        raise NotImplementedError('Generative neural network is far too slow.')
        pass

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return GNN().predict_proba((z[i],z[j]))

class igci(directed,unsigned):

    humanname = 'Information-geometric conditional independence'
    name = 'igci'
    labels = ['unsigned','unordered','infotheory','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return IGCI().predict_proba((z[i],z[j]))