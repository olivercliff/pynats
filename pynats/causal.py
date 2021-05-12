from cdt.causality.pairwise import ANM, BivariateFit, CDS, GNN, IGCI, RECI
from pynats.base import directed, undirected, parse_bivariate, unsigned

class anm(directed,unsigned):

    humanname = "Additive noise model"
    name = 'anm'
    labels = ['unsigned','model based','causal','unordered','linear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return ANM().predict_proba((z[i], z[j]))

class gpfit(directed,unsigned):
    
    humanname = 'Gaussian process bivariate fit'
    name = 'gpfit'
    labels = ['unsigned','model based','causal','unordered','normal','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return BivariateFit().b_fit_score(z[i], z[j])

class cds(directed,unsigned):
    
    humanname = 'Conditional distribution similarity statistic'
    name = 'cds'
    labels = ['unsigned','model based','causal','unordered','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return CDS().cds_score(z[i], z[j])

class gnn(undirected,unsigned):

    humanname = 'Shallow generative neural network'
    name = 'gnn'
    labels = ['unsigned','causal','unordered','neural network','nonlinear','undirected']

    def __init__(self):
        raise NotImplementedError('Having issues with this one.')
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

class reci(directed,unsigned):

    humanname = 'Neural correlation coefficient'
    name = 'reci'
    labels = ['unsigned','causal','unordered','neural network','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        return RECI().b_fit_score(z[i],z[j])