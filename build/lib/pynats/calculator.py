# Science/maths/computing tools
import numpy as np
import pandas as pd
import copy, yaml, importlib, time, warnings, os
from tqdm import tqdm
from collections import Counter
from copy import deepcopy

# From this package
from .data import Data
from .utils import convert_mdf_to_ddf

class Calculator():
    """Calculator for one multivariate time-series dataset
    """
    
    # Initializer / Instance Attributes
    def __init__(self,dataset=None,name=None,labels=None,configfile=os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'):

        self._load_yaml(configfile)

        duplicates = [name for name, count in Counter(self._measure_names).items() if count > 1]
        if len(duplicates) > 0:
            raise ValueError(f'Duplicate measure identifiers: {duplicates}.\n Check the config file for duplicates.')

        self._nmeasures = len(self._measures)
        self._nclasses = len(self._classes)
        self._proctimes = np.empty(self._nmeasures)
        self._name = name
        self._labels = labels

        print("Number of bivariate statistics: {}".format(self._nmeasures))

        if dataset is not None:
            self.load_dataset(dataset)

    @property
    def n_measures(self):
        return self._nmeasures

    @n_measures.setter
    def n_measures(self,n):
        raise Exception('Do not set this property externally.')

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self,d):
        raise Exception('Do not set this property externally. Use the load_dataset() method.')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,n):
        self._name = n

    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self,ls):
        self._labels = ls

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self,a):
        raise Exception('Do not set this property externally. Use the compute() method.')

    @property
    def group(self):
        try:
            return self._group
        except AttributeError as err:
            warnings.warn('Group undefined. Call set_group() method first.')
            raise AttributeError(err)

    @group.setter
    def group(self,g):
        raise Exception('Do not set this property externally. Use the set_group() method.')

    @property
    def group_name(self):
        try:
            return self._group_name
        except AttributeError as err:
            print(f'Group name undefined. Call set_group() method first.')
            return None

    @group_name.setter
    def group_name(self,g):
        raise Exception('Do not set this property externally. Use the group() method.')

    def _load_yaml(self,document):
        print("Loading configuration file: {}".format(document))
        self._classes = []
        self._class_names = []

        self._measures = []
        self._measure_names = []

        with open(document) as f:
            yf = yaml.load(f,Loader=yaml.FullLoader)

            # Instantiate the calc classes 
            for module_name in yf:
                print("*** Importing module {}".format(module_name))
                classes = yf[module_name]
                module = importlib.import_module(module_name,__package__)

                for class_name in classes:
                    paramlist = classes[class_name]

                    self._classes.append(getattr(module, class_name))
                    self._class_names.append(class_name)
                    if paramlist is not None:
                        for params in paramlist:
                            print(f'[{len(self._measures)}] Adding measure {module_name}.{class_name}(x,y,{params})...')
                            self._measures.append(self._classes[-1](**params))
                            self._measure_names.append(self._measures[-1].name)
                            print('Succesfully initialised with identifier "{}"'.format(self._measures[-1].name))
                    else:
                        print(f'[{len(self._measures)}] Adding measure {module_name}.{class_name}(x,y)...')
                        self._measures.append(self._classes[-1]())
                        self._measure_names.append(self._measures[-1].name)
                        print('Succesfully initialised with identifier "{}"'.format(self._measures[-1].name))

    def load_dataset(self,dataset):
        if not isinstance(dataset,Data):
            self._dataset = Data.convert_to_numpy(dataset)
        else:
            self._dataset = dataset
        self._adjacency = np.full((self._nmeasures,
                                    self.dataset.n_processes,
                                    self.dataset.n_processes), np.NaN)

    def compute(self,replication=None):
        """ Compute the dependency measures for all target processes for a given replication
        """
        if not hasattr(self,'_dataset'):
            raise AttributeError('Dataset not loaded yet. Please initialise with load_dataset.')

        if replication is None:
            replication = 0

        pbar = tqdm(range(self._nmeasures))
        for m in pbar:
            pbar.set_description(f'Processing [{self._name}: {self._measure_names[m]}]')
            start_time = time.time()
            try:
                self._adjacency[m] = self._measures[m].adjacency(self.dataset)
            except Exception as err:
                warnings.warn(f'Caught {type(err)} for measure "{self._measure_names[m]}": {err}')
                self._adjacency[m] = np.NaN
            self._proctimes[m] = time.time() - start_time
        pbar.close()

    def prune(self,meas_nans=0.2,proc_nans=0.8):
        """Prune the bad processes/measures
        """
        print(f'Pruning:\n\t- Measures with more than {100*meas_nans}% bad values'
                f', and\n\t- Processes with more than {100*proc_nans}% bad values')

        # First, iterate through the time-series and remove any that have NaN's > ts_nans
        M = self._nmeasures * (2*(self._dataset.n_processes-1))
        threshold = M * proc_nans
        rm_list = []
        for proc in range(self._dataset.n_processes):

            other_procs = [i for i in range(self._dataset.n_processes) if i != proc]

            flat_adj = self._adjacency[:,other_procs,proc].reshape((M//2,1))
            flat_adj = np.concatenate((flat_adj,self._adjacency[:,proc,other_procs].reshape((M//2,1))))

            nzs = np.count_nonzero(np.isnan(flat_adj))
            if nzs > threshold:
                # print(f'Removing process {proc} with {nzs} ({100*nzs/M.1f}%) special characters.')
                print(f'Removing process {proc} with {nzs} ({100*nzs/M}:1f%) special characters.')
                rm_list.append(proc)

        # Remove from the dataset object
        self._dataset.remove_process(rm_list)

        # Remove from the adjacency matrix (should probs move this to an attribute that cannot be set)
        self._adjacency = np.delete(self._adjacency,rm_list,axis=1)
        self._adjacency = np.delete(self._adjacency,rm_list,axis=2)

        # Then, iterate through the measures and remove any that have NaN's > meas_nans
        M = self._dataset.n_processes ** 2 - self._dataset.n_processes
        threshold = M * meas_nans
        il = np.tril_indices(self._dataset.n_processes,-1)

        rm_list = []
        for meas in range(self._nmeasures):

            flat_adj = self._adjacency[meas,il[1],il[0]].reshape((M//2,1))
            flat_adj = np.concatenate((flat_adj,
                                        self._adjacency[meas,il[0],il[1]].reshape((M//2,1))))

            # Ensure normalisation, etc., can happen
            if not np.isfinite(flat_adj.sum()):
                rm_list.append(meas)
                print(f'Measure "[{meas}] {self._measure_names[meas]}" has non-finite sum. Removing.')
                continue

            nzs = np.size(flat_adj) - np.count_nonzero(np.isfinite(flat_adj))
            if nzs > threshold:
                rm_list.append(meas)
                print(f'Removing measure "[{meas}] {self._measure_names[meas]}" with {nzs} ({100*nzs/M:.1f}%) '
                        f'NaNs (max is {threshold} [{100*meas_nans}%])')

        # Remove the measure from the adjacency and process times matrix
        self._adjacency = np.delete(self._adjacency,rm_list,axis=0)
        self._proctimes = np.delete(self._proctimes,rm_list,axis=0)

        # Remove from the measure lists (move to a method and protect measure)
        for meas in sorted(rm_list,reverse=True):
            del self._measures[meas]
            del self._measure_names[meas]

        self._nmeasures = len(self._measures)
        print('Number of statistics after pruning: {}'.format(self._nmeasures))

    def debias(self):
        """ Iterate through all measures and zero the unsigned measures (fixes absolute value errors when correlating)
        """
        for adj, m in zip(self._adjacency,self._measures):
            if not m.issigned():
                adj -= np.nanmin(adj)

    def set_group(self,classes):
        self._group = None
        self._group_name = None

        # Ensure this is a list of lists
        for i, cls in enumerate(classes):
            if not isinstance(cls,list):
                classes[i] = [cls]

        for i, i_cls in enumerate(classes):
            for j, j_cls in enumerate(classes):
                if i == j:
                    continue
                assert not set(i_cls).issubset(set(j_cls)), (f'Class {i_cls} is a subset of class {j_cls}.')

        labset = set(self.labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        if np.count_nonzero(matches) > 1:
            warnings.warn(f'More than one match for classes {classes}')
        else:
            try:
                id = np.where(matches)[0][0]
                self._group = id
                self._group_name = ', '.join(classes[id])
            except (TypeError,IndexError):
                pass

    # Merge two calculators (to include additional statistics)
    def merge(self,other):
        raise NotImplementedError()
        if self.name is not other.name:
            raise TypeError(f'Calculator name does do not match. Aborting merge.')
        
        for attr in ['name','n_processes','n_observations']:
            selfattr = getattr(self.dataset,attr)
            otherattr = getattr(other.dataset,attr)
            if selfattr is not otherattr:
                raise TypeError(f'Attribute {attr} does not match between calculators ({selfattr} != {otherattr})')

    def flatten(self,transformer=None):
        """ Gives a measure-by-edges matrix for correlations, etc.
        """
        M = self.dataset.n_processes
        n_edges = M*(M-1)

        il = np.tril_indices(M,-1)

        flatmat = np.empty((n_edges,self.n_measures))
        for f, adj in enumerate(self.adjacency):
            flatmat[:-1:2,f] = adj[il[1],il[0]]
            flatmat[1::2,f] = adj[il[0],il[1]]

        if transformer is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    flatmat = transformer.fit_transform(flatmat)
            except ValueError as err:
                print(f'Something went from with the transformer: {err}')

        edges = [None] * n_edges
        edges[:-1:2] = [f'{i}->{j}' for i, j in zip(*il)]
        edges[1::2] = [f'{j}->{i}' for i, j in zip(*il)]

        df = pd.DataFrame(flatmat, index=edges, columns=self._measure_names)
        df.columns.name = 'Bivariate statistic'
        df.index.name = 'Edges'
        return df

    def get_measure_labels(self):
        return { m.name : m.labels for m in self._measures }

    def get_correlation_df(self,with_labels=False,debias=False,which_measure=['spearman'],flatten_kwargs={}):
        # Sorts out pesky numerical issues in the unsigned measures
        if debias:
            self.debias()

        # Flatten (get edge-by-measure matrix)
        edges = self.flatten(**flatten_kwargs).abs()

        # Correlate the edge matrix (using pearson and/or spearman correlation)
        mdf = pd.DataFrame()
        if 'pearson' in which_measure:
            pmat = edges.corr(method='pearson')
            pmat.index = pd.MultiIndex.from_tuples([('pearson',m) for m in pmat.index],names=['Type','Source measure'])
            pmat.columns.name = 'Target measure'
            mdf = pmat
        if 'spearman' in which_measure:
            spmat = edges.corr(method='spearman')
            spmat.index = pd.MultiIndex.from_tuples([('spearman',m) for m in spmat.index],names=['Type','Source measure'])
            spmat.columns.name = 'Target measure'
            mdf = mdf.append(spmat)

        if with_labels:
            return mdf, self.get_measure_labels()
        else:
            return mdf
    
    def communities(self,flatten_kwargs={}):
        if not hasattr(self,'_mdf'):
            self._mdf = pd.DataFrame()
            for calc in [c[0] for c in self.calculators.values]:
                corrmat = calc.correlation_matrix(**flatten_kwargs)

                # Adds another hierarchical level giving the dataset name
                df2 = pd.concat({calc.name: corrmat}, names=['Dataset']) 
                self._mdf = self._mdf.append(df2)
        return self._mdf

""" CalculatorFrame
Container for batch level commands, like computing/pruning/initialising multiple datasets at once
"""
def forall(func):
    def do(self,*args,**kwargs):
        try:
            for i in self._calculators.index:
                calc_ser = self._calculators.loc[i]
                for calc in calc_ser:
                    func(calc,*args,**kwargs)
        except AttributeError:
            raise AttributeError(f'No calculators in frame yet. Initialise before calling {func}')
    return do

class CalculatorFrame():

    def __init__(self,calculators=None,name=None,datasets=None,names=None,labels=None,**kwargs):
        if calculators is not None:
            self.set_calculator(calculators)

        self.name = name

        if datasets is not None:
            if names is None:
                names = [None] * len(datasets)
            if labels is None:
                labels = [None] * len(datasets)
            self.init_from_list(datasets,names,labels,**kwargs)

    @property
    def name(self):
        if hasattr(self,'_name') and self._name is not None:
            return self._name
        else:
            return ''

    @name.setter
    def name(self,n):
        self._name = n

    @staticmethod
    def from_calculator(calculator):
        cf = CalculatorFrame()
        cf.add_calculator(calculator)
        return cf

    def set_calculator(self,calculators):
        if hasattr(self, '_dataset'):
            Warning('Overwriting dataset without explicitly deleting.')
            del(self._calculators)

        if isinstance(calculators,Calculator):
            calculators = [calculators]

        for calc in calculators:
            self.add_calculator(calc)
    
    def add_calculator(self,calc):

        if not hasattr(self,'_calculators'):
            self._calculators = pd.DataFrame()

        if isinstance(calc,CalculatorFrame):
            self._calculators = pd.concat([self._calculators,calc])
        elif isinstance(calc,Calculator):
            self._calculators = self._calculators.append(pd.Series(data=calc,name=calc.name),ignore_index=True)
        elif isinstance(calc,pd.DataFrame):
            if isinstance(calc.iloc[0],Calculator):
                self._calculators = calc
            else:
                raise TypeError('Received dataframe but it is not in known format.')
        else:
            raise TypeError(f'Unknown data type: {type(calc)}.')

        self.n_calculators = len(self.calculators.index)
    
    def init_from_list(self,datasets,names,labels,**kwargs):
        base_calc = Calculator(**kwargs)
        for i, dataset in enumerate(datasets):
            calc = copy.deepcopy(base_calc)
            calc.load_dataset(dataset)
            calc.name = names[i]
            calc.labels = labels[i]
            self.add_calculator(calc)

    def init_from_yaml(self,document,normalise=True,n_processes=None,n_observations=None,**kwargs):
        datasets = []
        names = []
        labels = []
        with open(document) as f:
            yf = yaml.load(f,Loader=yaml.FullLoader)

            for config in yf:
                try:
                    file = config['file']
                    dim_order = config['dim_order']
                    names.append(config['name'])
                    labels.append(config['labels'])
                    datasets.append(Data(data=file,dim_order=dim_order,name=names[-1],normalise=normalise,n_processes=n_processes,n_observations=n_observations))
                except Exception as err:
                    print(f'Loading dataset: {config} failed ({err}).')

        self.init_from_list(datasets,names,labels,**kwargs)

    @property
    def calculators(self):
        """Return data array."""
        try:
            return self._calculators
        except AttributeError:
            return None

    @calculators.setter
    def calculators(self, cs):
        if hasattr(self, 'calculators'):
            raise AttributeError('You can not assign a value to this attribute'
                                 ' directly, use the set_data method instead.')
        else:
            self._calculators = cs

    @calculators.deleter
    def calculators(self):
        print('Overwriting existing calculators.')
        del(self._calculators)

    def merge(self,other):
        try:
            self._calculators = self._calculators.append(other._calculators,ignore_index=True)
        except AttributeError:
            self._calculators = other._calculators

    @forall
    def compute(calc):
        calc.compute()

    @forall
    def prune(calc,**kwargs):
        calc.prune(**kwargs)

    @forall
    def set_group(calc,*args):
        calc.set_group(*args)

    @forall
    def debias(calc):
        calc.debias()

    def flattenall(self,**kwargs):
        df = pd.DataFrame()
        for i in self.calculators.index:
            calc = self.calculators.loc[i][0]
            df2 = calc.flatten(**kwargs)
            df = df.append(df2, ignore_index=True)

        df.dropna(axis='index',how='all',inplace=True)
        return df

    def get_correlation_df(self,with_labels=False,flatten_kwargs={},**kwargs):
        if with_labels:
            mlabels = {}
            dlabels = {}

        mdf = pd.DataFrame()
        for calc in [c[0] for c in self.calculators.values]:
            out = calc.get_correlation_df(with_labels=with_labels,flatten_kwargs=flatten_kwargs,**kwargs)

            if with_labels:
                df2 = pd.concat({calc.name: out[0]}, names=['Dataset']) 
                mlabels = mlabels | out[1]
                dlabels[calc.name] = calc.labels
            else:
                df2 = pd.concat({calc.name: out}, names=['Dataset']) 

            # Adds another hierarchical level giving the dataset name
            mdf = mdf.append(df2)

        if with_labels:
            return mdf, mlabels, dlabels
        else:
            return mdf

class CorrelationFrame():

    def __init__(self,cf=None,flatten_kwargs={},**kwargs):
        self._mlabels = {}
        self._dlabels = {}
        self._mdf = pd.DataFrame()
        
        if cf is not None:
            if isinstance(cf,CalculatorFrame) or isinstance(cf,Calculator):
                cf = CalculatorFrame(cf)
                # Store the measure-focused dataframe, measure labels, and dataset labels
                self._mdf, self._mlabels, self._dlabels = cf.get_correlation_df(with_labels=True,flatten_kwargs=flatten_kwargs,**kwargs)
                self._name = cf.name
            else:
                self.merge(cf)

    @property
    def name(self):
        if not hasattr(self,'_name'):
            return ''
        else:
            return self._name

    @name.setter
    def name(self,n):
        self._name = n

    @property
    def mdf(self):
        return self._mdf

    @property
    def ddf(self):
        if not hasattr(self,'_ddf'):
            self._ddf = convert_mdf_to_ddf(self.mdf)
        return self._ddf
    
    @property
    def n_datasets(self):
        return self.ddf.shape[1]

    @property
    def n_measures(self):
        return self.mdf.shape[1]

    @property
    def mlabels(self):
        return self._mlabels

    @property
    def dlabels(self):
        return self._dlabels

    @mdf.setter
    def mdf(self):
        raise AttributeError('Do not directly set the mdf attribute.')

    @mlabels.setter
    def mlabels(self):
        raise AttributeError('Do not directly set the mlabels attribute.')
        
    @dlabels.setter
    def dlabels(self):
        raise AttributeError('Do not directly set the dlabels attribute.')

    def merge(self,other):
        self._mdf = self._mdf.append(other.mdf)
        self._mlabels = self._mlabels | other.mlabels
        self._dlabels = self._dlabels | other.dlabels

        # Make sure to re-run this otherwise we'll have the old one
        self._ddf = convert_mdf_to_ddf(self.mdf)

    def get_feature_matrix(self,mthresh=0.8,dthresh=0.8):
        fm = self.ddf.drop_duplicates()
        fm = fm.dropna(axis=0,thresh=mthresh*fm.shape[1])
        fm = fm.dropna(axis=1,thresh=dthresh*fm.shape[0])
        return fm

    @staticmethod
    def _verify_classes(classes):
        # Ensure this is a list of lists
        for i, cls in enumerate(classes):
            if not isinstance(cls,list):
                classes[i] = [cls]

        for i, i_cls in enumerate(classes):
            for j, j_cls in enumerate(classes):
                if i == j:
                    continue
                assert not set(i_cls).issubset(set(j_cls)), (f'Class {i_cls} is a subset of class {j_cls}.')

    @staticmethod
    def _get_group(labels,classes):
        labset = set(labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        # Iterate through all 
        if np.count_nonzero(matches) > 1:
            warnings.warn(f'More than one match for classes {classes}')
        else:
            try:
                myid = np.where(matches)[0][0]
                return myid
            except (TypeError,IndexError):
                return -1

    @staticmethod
    def _set_groups(classes,labels,group_names,group):
        CorrelationFrame._verify_classes(classes)
        for m in labels:
            group[m] = CorrelationFrame._get_group(labels[m],classes)

    def set_mgroups(self,classes):
        # Initialise the classes
        self._mgroup_names = { i : ', '.join(c) for i, c in enumerate(classes) }
        self._mgroup_names[-1] = 'N/A'

        self._mgroup_ids = { m : -1 for m in self._mlabels }
        CorrelationFrame._set_groups(classes,self._mlabels,self._mgroup_names,self._mgroup_ids)
            

    def set_dgroups(self,classes):
        self._dgroup_names = { i : ', '.join(c) for i, c in enumerate(classes) }
        self._dgroup_names[-1] = 'N/A'

        self._dgroup_ids = { d : -1 for d in self._dlabels }
        CorrelationFrame._set_groups(classes,self._dlabels,self._dgroup_names,self._dgroup_ids)

    def get_dgroup_ids(self,names=None):
        if names is None:
            names = self._ddf.columns
        return [self._dgroup_ids[n] for n in names]

    def get_dgroup_names(self,names=None):
        if names is None:
            names = self._ddf.columns
        return [self._dgroup_names[i] for i in self.get_dgroup_ids(names)]

    def get_mgroup_ids(self,names=None):
        if names is None:
            names = self._mdf.columns
        return [self._mgroup_ids[n] for n in names]

    def get_mgroup_names(self,names=None):
        if names is None:
            names = self._mdf.columns
        return [self._mgroup_names[i] for i in self.get_mgroup_ids(names)]