import torch
import numpy as np
from numpy.typing import NDArray
import copy
import random

rng = np.random.default_rng()

class DataGenerator(torch.utils.data.Dataset):

    def __init__(self,
                 batch_size: int,
                 flux: NDArray,
                 ivar: NDArray,
                 continuum: NDArray,
                 #flux_n: NDArray,
                 #ivar_n: NDArray
        ):

        self.batch_size = batch_size

        # get rid of nans if they exist
        nan_mask = (np.sum(~np.isnan(flux),axis=-1).astype('bool'))*(np.sum(~np.isnan(ivar),axis=-1).astype('bool'))*(np.sum(~np.isnan(continuum),axis=-1).astype('bool'))
        self.flux = copy.deepcopy(flux)
        self.ivar = copy.deepcopy(ivar)
        
        self.flux_n = copy.deepcopy(flux/continuum)
        self.ivar_n = copy.deepcopy(ivar * continuum**2)

        self.data_length = len(self.flux_n)
        self.steps_per_epoch = self.data_length // self.batch_size
        self.idx_list = np.arange(self.data_length)

        self.epoch_flux_n = None
        self.epoch_ivar_n = None

        self.epoch_end()

    def __iter__(self):
        for i in range(len(self)):
            tmp_idxs = self.idx_list[i*self.batch_size:(i+1)*self.batch_size]
            yield (self.epoch_flux_n[tmp_idxs],self.epoch_ivar_n[tmp_idxs])

    def __len__(self):
        return self.steps_per_epoch
    
    def epoch_end(self):
        tmp = list(zip(self.flux_n,self.ivar_n))
        tmp = np.random.permutation(self.data_length)
        self.epoch_flux_n,self.epoch_ivar_n = self.flux_n[tmp],self.ivar_n[tmp]
        
