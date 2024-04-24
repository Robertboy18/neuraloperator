import torch
import tensorly as tl
from tensorly import tucker_to_tensor
from tensorly.decomposition import tucker

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.factors = None
        self.default = False
        
    def project(self, full_rank_grad, iter):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix, self.factors = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')     
        return self.ortho_matrix # this is where I need to knnow what factors to multiply 

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = tl.tucker_to_tensor((self.ortho_matrix, self.factors))        
        return full_rank_grad * self.scale
        
        
    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights
        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        rank1 = rank
        core, factors = tucker(matrix, rank=[rank1, rank1, *matrix.shape[2:]])
        return core, factors


    def get_orthogonal_matrix1(self, weights, rank, type):
        #print("original", weights.shape)
        #zif len(weights.shape) == 2:
        module_params = weights[..., -2:, -1:].squeeze()
        #print("weights matrix size", weights.shape)
        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
        #print("SHAPEEEEEE", matrix.shape)
        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
        
        #make the smaller matrix always to be orthogonal matrix
        if type=='right':
            A = U[:, :rank] @ torch.diag(s[:rank])
            B = Vh[:rank, :]
            
            if not float_data:
                B = B.to(original_device).type(original_type)
            print("B", B.shape)
            module_params[..., -2:, -1:] = B
            return module_params
        elif type=='left':
            A = U[:, :rank]
            print(torch.diag(s[:rank]) @ Vh[:rank, :])
            B = torch.diag(s[:rank]) @ Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')