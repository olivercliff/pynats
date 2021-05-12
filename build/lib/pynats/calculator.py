# Science/maths/computing tools
import numpy as np
import pandas as pd
import copy
import yaml
import importlib
import time
import warnings
import os

# Plotting tools
from tqdm import tqdm
from tqdm import trange
from collections import Counter

# From this package
from .data import Data

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

        print("Number of pairwise measures: {}".format(self._nmeasures))

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
        raise Exception('Do not set this property externally. Use the load_dataset method.')

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
        raise Exception('Do not set this property externally. Use the compute method to obtain property.')

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

    def prune(self,meas_nans=0.0,proc_nans=0.9):
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
                print('Removing process {} with {} ({}.1f%) special characters.'.format(proc,nzs,100*nzs/M))
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
                print('Removing measure "[{}] {}" with {} ({:.1f}%) '
                        'NaNs (max is {} [{}%])'.format(meas, self._measure_names[meas],
                                                        nzs,100*nzs/M, threshold, 100*meas_nans))

        # Remove the measure from the adjacency and process times matrix
        self._adjacency = np.delete(self._adjacency,rm_list,axis=0)
        self._proctimes = np.delete(self._proctimes,rm_list,axis=0)

        # Remove from the measure lists (move to a method and protect measure)
        for meas in sorted(rm_list,reverse=True):
            del self._measures[meas]
            del self._measure_names[meas]

        self._nmeasures = len(self._measures)
        print('Number of pairwise measures after pruning: {}'.format(self._nmeasures))

    # TODO - merge two calculators (e.g., to include missing/decentralised data or measures)
    def merge(self,other):
        raise NotImplementedError

    def save(self,filename):
        raise NotImplementedError

def forall(func):
    def do(self,**kwargs):
        try:
            for i in self._calculators.index:
                calc_ser = self._calculators.loc[i]
                for calc in calc_ser:
                    func(self,calc,**kwargs)
        except AttributeError:
            raise AttributeError(f'No calculators in frame yet. Initialise before calling {func}')
    return do

class CalculatorFrame():

    def __init__(self,name=None,datasets=None,names=None,labels=None,calculators=None,**kwargs):
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

    def set_calculator(self,calculators,names=None):
        if hasattr(self, '_dataset'):
            Warning('Overwriting dataset without explicitly deleting.')
            del(self._calculators)

        for calc, i in calculators:
            if names is not None:
                self.add_calculator(calc,names[i])
            else:
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
            calc.label = labels[i]
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


    @forall
    def compute(self,calc):
        calc.compute()

    @forall
    def prune(self,calc,**kwargs):
        calc.prune(**kwargs)