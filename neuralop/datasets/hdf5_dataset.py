import torch
import h5py
from torch.utils.data import Dataset
import numpy as np

class H5pyDataset(Dataset):
    """PDE h5py dataset"""
    def __init__(self, data_path, resolution=128, transform_x=None, transform_y=None,
                 n_samples=None, mode = 'train'):
        resolution_to_step = {128:8, 256:4, 512:2, 1024:1}
        try:
            subsample_step = resolution_to_step[resolution]
        except KeyError:
            raise ValueError(f'Got {resolution=}, expected one of {resolution_to_step.keys()}')

        self.subsample_step = subsample_step
        self.data_path = data_path
        self._data = None
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.mode = mode
        if n_samples is not None:
            self.n_samples = n_samples
        else:
            with h5py.File(str(self.data_path), 'r') as f:
                self.n_samples = f['x'].shape[0]
                print("N_SAMPLES: ", self.n_samples)

    @property
    def data(self):
        if self._data is None:
            if self.mode == 'train':
                self._data = h5py.File(str(self.data_path), 'r')
        return self._data

    def _attribute(self, variable, name):
        return self.data[variable].attrs[name]

    def __len__(self):
        return self.n_samples-2
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            assert idx < self.n_samples, f'Trying to access sample {idx} of dataset with {self.n_samples} samples'
        else:
            for i in idx:
                assert i < self.n_samples, f'Trying to access sample {i} of dataset with {self.n_samples} samples'
        
        #print(self.data.keys())
        #print(self.data['512x512x2_wn1.0']['fields'])64x64x2_wn1.0
        self.subsample_step = 1
        x = self.data['512x512x2_wn16.0']['fields'][idx, 1:, ::self.subsample_step, ::self.subsample_step]
        #x2= self.data['512x512x2_wn16.0']['fields'][idx+1, 1:, ::self.subsample_step, ::self.subsample_step]
        #x3= self.data['512x512x2_wn16.0']['fields'][idx+2, 1:, ::self.subsample_step, ::self.subsample_step]
        """
        x4= self.data['512x512x2_wn16.0']['fields'][idx+3, 1:, ::self.subsample_step, ::self.subsample_step]
        x5= self.data['512x512x2_wn16.0']['fields'][idx+4, 1:, ::self.subsample_step, ::self.subsample_step]
        x6= self.data['512x512x2_wn16.0']['fields'][idx+5, 1:, ::self.subsample_step, ::self.subsample_step]
        x7= self.data['512x512x2_wn16.0']['fields'][idx+6, 1:, ::self.subsample_step, ::self.subsample_step]
        x8= self.data['512x512x2_wn16.0']['fields'][idx+7, 1:, ::self.subsample_step, ::self.subsample_step]
        x9= self.data['512x512x2_wn16.0']['fields'][idx+8, 1:, ::self.subsample_step, ::self.subsample_step]
        x10= self.data['512x512x2_wn16.0']['fields'][idx+9, 1:, ::self.subsample_step, ::self.subsample_step]
        x11= self.data['512x512x2_wn16.0']['fields'][idx+10, 1:, ::self.subsample_step, ::self.subsample_step]
        x12= self.data['512x512x2_wn16.0']['fields'][idx+11, 1:, ::self.subsample_step, ::self.subsample_step]
        x13= self.data['512x512x2_wn16.0']['fields'][idx+12, 1:, ::self.subsample_step, ::self.subsample_step]
        x14= self.data['512x512x2_wn16.0']['fields'][idx+13, 1:, ::self.subsample_step, ::self.subsample_step]
        x15= self.data['512x512x2_wn16.0']['fields'][idx+14, 1:, ::self.subsample_step, ::self.subsample_step]
        #print(x1.shape, x2.shape, x3.shape)
        """
        #x = np.concatenate((x1,x2,x3))
        y = self.data['512x512x2_wn16.0']['fields'][idx+1, 1:, ::self.subsample_step, ::self.subsample_step]
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        #print("x shape: ", x.shape)
        #print("y shape: ", y.shape)

        if self.transform_x:
            x = self.transform_x(x)
            #pass

        if self.transform_y:
            y = self.transform_y(y)
            #pass

        return {'x': x, 'y': y}
    
