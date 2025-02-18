import torch
import numpy as np
import h5py
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using %s' % device)

if device == 'cpu':
    torch.set_num_threads(176);
print('Number of threads', torch.get_num_threads())

exit()

lr = 1e-4
lr_min = 1e-10
batch_size = 1024
epochs = 5000
cos_anneal_t0 = 500
check_every_n_epochs = 500
num_data = int(1e7)

sys.path.append('..')
from xp_vae.model import ScatterVAE
model = ScatterVAE().to(device)
model.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

BOSS_NORM_PATH = '/home/way/MDwarf_Continuum/mdwarf_contin/M_dwarf_spec_coadd.h5'
f = h5py.File(BOSS_NORM_PATH,'r')

#g_flux = f['phot_g_mean_flux'][:,np.newaxis]
#xp = f['coeffs']/g_flux
#xp_err = f['coeff_errs']/g_flux
flux = f['flux'][:,478:]
ivar = f['ivar'][:,478:]
continuum = f['continuum'][:,478:]

nan_mask = (np.sum(~np.isnan(flux),axis=-1).astype('bool'))*(np.sum(~np.isnan(ivar),axis=-1).astype('bool'))*(np.sum(~np.isnan(continuum),axis=-1).astype('bool'))
flux = flux[nan_mask]
ivar = ivar[nan_mask]
continuum = continuum[nan_mask]

#norm = np.load('../data/apogee_norm.npz')
#xp = (xp - norm['mu']) / norm['sig']
#xp_err = xp_err / norm['sig']

lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                      T_0=cos_anneal_t0,
                                                                                      T_mult=1,
                                                                                      eta_min=lr_min,
                                                                                      last_epoch=-1,
                                                                                    )

model.fit(flux, ivar, continuum,
          epochs=epochs,
          lr_scheduler=lr_scheduler,
          batch_size=batch_size,
          checkpoint_every_n_epochs=check_every_n_epochs,
          output_direc='../models/BOSS_MODEL')

model.save('../models/BOSS_MODEL')
