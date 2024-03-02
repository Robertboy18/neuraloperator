import torch
from pathlib import Path

#from .output_encoder import UnitGaussianNormalizer
from neuralop.utils import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding2D
from .data_transforms import DefaultDataProcessor
import scipy.io

# from .hdf5_dataset import H5pyDataset

# def load_navier_stokes_hdf5(data_path, n_train, batch_size,
#                             train_resolution=128,
#                             test_resolutions=[128, 256, 512, 1024],
#                             n_tests=[2000, 500, 500, 500],
#                             test_batch_sizes=[8, 4, 1],
#                             positional_encoding=True,
#                             grid_boundaries=[[0,1],[0,1]],
#                             encode_input=True,
#                             encode_output=True,
#                             num_workers=0, pin_memory=True, persistent_workers=False):
#     data_path = Path(data_path)

#     training_db = H5pyDataset(data_path / 'navier_stokes_1024_train.hdf5', n_samples=n_train, resolution=train_resolution)
#     in_normalizer = None
#     out_normalizer = None
#     pos_encoding = None

#     if encode_input:
#         x_mean = training_db._attribute('x', 'mean')
#         x_std = training_db._attribute('x', 'std')
        
#         in_normalizer = Normalizer(x_mean, x_std)
    
#     if positional_encoding:
#         pos_encoding = PositionalEmbedding2D(grid_boundaries)
    
#     if encode_output:
#         y_mean = training_db._attribute('y', 'mean')
#         y_std = training_db._attribute('y', 'std')
        
#         out_normalizer = Normalizer(y_mean, y_std)

#     data_processor = DefaultDataProcessor(in_normalizer=in_normalizer,
#                                           out_normalizer=out_normalizer,
#                                           positional_encoding=pos_encoding)
    
#     train_loader = torch.utils.data.DataLoader(training_db,
#                                                batch_size=batch_size, 
#                                                shuffle=True,
#                                                num_workers=num_workers,
#                                                pin_memory=pin_memory,
#                                                persistent_workers=persistent_workers)

#     test_loaders = dict()
#     for (res, n_test, test_batch_size) in zip(test_resolutions, n_tests, test_batch_sizes):
#         print(f'Loading test db at resolution {res} with {n_test} samples and batch-size={test_batch_size}')

#         test_db = H5pyDataset(data_path / 'navier_stokes_1024_test.hdf5', n_samples=n_test, resolution=res)
    
#         test_loaders[res] = torch.utils.data.DataLoader(test_db, 
#                                                         batch_size=test_batch_size,
#                                                         shuffle=False,
#                                                         num_workers=num_workers, 
#                                                         pin_memory=pin_memory, 
#                                                         persistent_workers=persistent_workers)

#     return train_loader, test_loaders, data_processor


def load_navier_stokes_pt(data_path, train_resolution,
                          n_train, n_tests,
                          batch_size, test_batch_sizes,
                          test_resolutions,
                          grid_boundaries=[[0,1],[0,1]],
                          positional_encoding=True,
                          encode_input=True,
                          encode_output=True,
                          encoding='channel-wise',
                          channel_dim=1,
                          num_workers=2,
                          pin_memory=True, 
                          persistent_workers=True,
                          ):
    """Load the Navier-Stokes dataset
    """
    #assert train_resolution == 128, 'Loading from pt only supported for train_resolution of 128'

    train_resolution_str = str(train_resolution)

    data = torch.load(Path(data_path).joinpath('nsforcing_' + train_resolution_str + '_train.pt').as_posix())
    x_train = data['x'][0:n_train, :, :].unsqueeze(channel_dim).clone()
    y_train = data['y'][0:n_train, :, :].unsqueeze(channel_dim).clone()
    del data

    idx = test_resolutions.index(train_resolution)
    test_resolutions.pop(idx)
    n_test = n_tests.pop(idx)
    test_batch_size = test_batch_sizes.pop(idx)

    data = torch.load(Path(data_path).joinpath('nsforcing_' + train_resolution_str + '_test.pt').as_posix())
    x_test = data['x'][:n_test, :, :].unsqueeze(channel_dim).clone()
    y_test = data['y'][:n_test, :, :].unsqueeze(channel_dim).clone()
    print(x_train.shape)
    del data
    
    pos_encoding = None

    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
    else:
        input_encoder = None

    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
    else:
        output_encoder = None
    
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries)

    data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                          out_normalizer=output_encoder,
                                          positional_encoding=pos_encoding)
    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db,
                                               batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    test_loaders =  {train_resolution: test_loader}
    for (res, n_test, test_batch_size) in zip(test_resolutions, n_tests, test_batch_sizes):
        print(f'Loading test db at resolution {res} with {n_test} samples and batch-size={test_batch_size}')
        x_test, y_test = _load_navier_stokes_test_HR(data_path, n_test, resolution=res, channel_dim=channel_dim)
        if input_encoder is not None:
            x_test = input_encoder.encode(x_test)

        test_db = TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_db,
                                                  batch_size=test_batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        test_loaders[res] = test_loader

    return train_loader, test_loaders, data_processor


def _load_navier_stokes_test_HR(data_path, n_test, resolution=256,
                                channel_dim=1,
                               ):
    """Load the Navier-Stokes dataset
    """
    if resolution == 128:
        downsample_factor = 8
    elif resolution == 256:
        downsample_factor = 4
    elif resolution == 512:
        downsample_factor = 2
    elif resolution == 1024:
        downsample_factor = 1
    else:
        raise ValueError(f'Invalid resolution, got {resolution}, expected one of [128, 256, 512, 1024].')
    
    data = torch.load(Path(data_path).joinpath('nsforcing_128_test.pt').as_posix())

    if not isinstance(n_test, int):
        n_samples = data['x'].shape[0]
        n_test = int(n_samples*n_test)
        
    x_test = data['x'][:n_test, ::downsample_factor, ::downsample_factor].unsqueeze(channel_dim).clone()
    y_test = data['y'][:n_test, ::downsample_factor, ::downsample_factor].unsqueeze(channel_dim).clone()
    del data

    return x_test, y_test


def load_ns_high(data_path, ntrain=8, ntest=2, subsampling_rate=1, batch_size=50, T = 5000, in_dim = 1, out_dim = 1, ntimeindex = 1, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True):
    
    data_train = torch.load(data_path)['vorticity']
    rate = subsampling_rate
    T_in = 1
    S = 256//subsampling_rate
    step = 1
    T1 = T//ntimeindex
    T_in = T_in
    T = T
    sub = subsampling_rate
    T_out = T_in + T
    ntimeindex = ntimeindex
    T1 = T1
    channel_dim = 1

    data = data_train[..., ::sub, ::sub]

    x_train = data[:ntrain, T_in-1:T_out-1:ntimeindex].clone()
    y_train = data[:ntrain, T_in:T_out:ntimeindex].clone()

    x_test = data[-ntest:, T_in-1:T_out-1:ntimeindex].clone()
    y_test = data[-ntest:, T_in:T_out:ntimeindex].clone()
    
    del data

    x_train = x_train.reshape(ntrain*T1, S, S).unsqueeze(channel_dim)
    y_train = y_train.reshape(ntrain*T1, S, S).unsqueeze(channel_dim)
    
    x_test = x_test.reshape(ntest*T1, S, S).unsqueeze(channel_dim)
    y_test = y_test.reshape(ntest*T1, S, S).unsqueeze(channel_dim)


    train_db = TensorDataset(x_train, y_train)
    test_db = TensorDataset(x_test, y_test)
    
    test_resolutions = [256]
    n_tests = [10000]
    test_batch_sizes = [50]
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = torch.utils.data.DataLoader(test_db,batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loaders =  {256: test_loader}
    for (res, n_test, test_batch_size) in zip(test_resolutions, n_tests, test_batch_sizes):
        print(f'Loading test db at resolution {res} with {n_test} samples and batch-size={test_batch_size}')
        x_test, y_test = x_test, y_test

        test_db = TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_db,
                                                  batch_size=test_batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        test_loaders[res] = test_loader

    data_processor = None
    return train_loader, test_loaders, data_processor

def load_ns_time(train_path, test_path, ntrain=1000, ntest=200, subsampling_rate=1, channel_dim = 1, batch_size=32, T = 10, time = False, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True):
        
    data_train = scipy.io.loadmat(train_path)
    data_test = scipy.io.loadmat(test_path)
    rate = subsampling_rate

    T_in = 10
    T = T
    S = 64//rate

    x_train = torch.tensor(data_train['u'], dtype=torch.float)[:ntrain, ::rate, ::rate, :T_in]
    y_train = torch.tensor(data_train['u'], dtype=torch.float)[:ntrain, ::rate, ::rate, T_in:T+T_in]#.unsqueeze(channel_dim)

    x_test = torch.tensor(data_test['u'], dtype=torch.float)[-ntest:, ::rate, ::rate,:T_in]
    y_test = torch.tensor(data_test['u'], dtype=torch.float)[-ntest:, ::rate, ::rate,T_in:T+T_in]#.unsqueeze(channel_dim)
    
    if time:
        x_train = x_train.reshape(ntrain, S, S, T_in)#.unsqueeze(channel_dim)
        x_test = x_test.reshape(ntest, S, S, T_in)#.unsqueeze(channel_dim)
    else:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

        x_train = x_train.reshape(ntrain, S, S, 1, T_in).repeat([1,1,1,T,1])
        x_test = x_test.reshape(ntest, S, S, 1, T_in).repeat([1,1,1,T,1])

    del data_train
    del data_test

    train_db = TensorDataset(x_train, y_train)
    test_db = TensorDataset(x_test, y_test)
    
    test_resolutions = [64]
    n_tests = [100]
    test_batch_sizes = [32]
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = torch.utils.data.DataLoader(test_db,batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loaders =  {64: test_loader}
    for (res, n_test, test_batch_size) in zip(test_resolutions, n_tests, test_batch_sizes):
        print(f'Loading test db at resolution {res} with {n_test} samples and batch-size={test_batch_size}')
        x_test, y_test = x_test, y_test

        test_db = TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_db,
                                                  batch_size=test_batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        test_loaders[res] = test_loader

    data_processor = None
    return train_loader, test_loaders, data_processor