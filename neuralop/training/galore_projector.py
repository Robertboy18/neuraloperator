import torch
from tensorly.decomposition import tucker
from tensorly import tenalg 

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.transformed_low_rank = None
        self.default = True
        
    def project(self, full_rank_grad, iter):
        if self.default == True:
            return full_rank_grad
        if self.ortho_matrix is None and iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank)    
        self.transformed_low_rank = self.transform(self.ortho_matrix, full_rank_grad)
        return self.transformed_low_rank

    def project_back(self, low_rank_grad):
        if self.default == True:
            return low_rank_grad
        else:
            full_rank_grad = self.inverse_transform(self.ortho_matrix, self.transformed_low_rank)     
            return full_rank_grad * self.scale
        
    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank1):
        module_params = weights
        if module_params.data.dtype != torch.float:
            matrix = module_params.data.float()
        else:
            matrix = module_params.data
        tucker_tensor = tucker(matrix, rank=rank1)
        return tucker_tensor

    def transform(self, tensor, x):
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, tensor, x):
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors)
