print("Importing packages...", flush=True)
import torch
import numpy as np
import h5py
import sys

print("Checking CUDA...", flush=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using %s' % device)

#if device == 'cpu':
#    torch.set_num_threads(176);
print('Number of threads', torch.get_num_threads(), flush=True)

lr = 1e-4
lr_min = 1e-10
batch_size = 1024
epochs = 10
cos_anneal_t0 = 500
check_every_n_epochs = 1
num_data = int(1e7)

print("Importing model...", flush=True)
sys.path.append('..')
from xp_vae.model import ScatterVAE
model = ScatterVAE().to(device)
model.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("Opening spectra...", flush=True)
BOSS_NORM_PATH = '/home/way/MDwarf_Continuum/mdwarf_contin/M_dwarf_spec_coadd.h5'
f = h5py.File(BOSS_NORM_PATH,'r')

print("Defining flux, ivar, continuum...", flush=True)
N_SPECTRA = 5000
flux = f['flux'][:N_SPECTRA,478:]
ivar = f['ivar'][:N_SPECTRA,478:]
continuum = f['continuum'][:N_SPECTRA,478:]

print("Removing nans...", flush=True)
nan_mask = (np.sum(~np.isnan(flux),axis=-1).astype('bool'))*(np.sum(~np.isnan(ivar),axis=-1).astype('bool'))*(np.sum(~np.isnan(continuum),axis=-1).astype('bool'))
flux = flux[nan_mask]
ivar = ivar[nan_mask]
continuum = continuum[nan_mask]

print("Remove spectra with crazy ivar...", flush=True)
z_ivar_mask = ~(np.sum((ivar==0), axis=-1)>4170/4)
flux = flux[z_ivar_mask]
ivar = ivar[z_ivar_mask]
continuum = continuum[z_ivar_mask]
print((~z_ivar_mask).sum())

flux_n = flux/continuum - 1
ivar_n = ivar*continuum**2

print("Infinite error on outlier flux...", flush=True)
outlier_mask = (flux_n - np.mean(flux_n) > np.quantile(flux_n - np.mean(flux_n), 0.99)) | \
                (flux_n - np.mean(flux_n) < np.quantile(flux_n - np.mean(flux_n), 0.01)) | \
                (ivar_n - np.mean(ivar_n) > np.quantile(ivar_n - np.mean(ivar_n), 0.99)) | \
                (ivar_n - np.mean(ivar_n) < np.quantile(ivar_n - np.mean(ivar_n), 0.01))
ivar_n[outlier_mask] = 0.
print(outlier_mask.sum())

print("Setting low ivar to zero...", flush=True)
low_mask = (ivar_n<10**-3)
ivar_n[low_mask] = 0.


print("Starting scheduler...", flush=True)
lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                      T_0=cos_anneal_t0,
                                                                                      T_mult=1,
                                                                                      eta_min=lr_min,
                                                                                      last_epoch=-1,
                                                                                    )

print("Fitting model...", flush=True)
model.fit(flux_n, ivar_n, #continuum,
          epochs=epochs,
          lr_scheduler=lr_scheduler,
          batch_size=batch_size,
          checkpoint_every_n_epochs=check_every_n_epochs,
          output_direc='../models/BOSS_MODEL_v2',
          terminate_on_nan=True)

model.save('../models/BOSS_MODEL_v2')
