import torch

def sketch_for_low_rank_approx(A, k, l, orthonormalize=True):
    """
    Implement Algorithm 1: Sketch for Low-Rank Approximation
    
    Args:
        A (torch.Tensor): Input matrix of size (m, n)
        k (int): Sketch size parameter for range sketch
        l (int): Sketch size parameter for co-range sketch
        
    Returns:
        Omega (torch.Tensor): Random test matrix of size (n, k)
        Psi (torch.Tensor): Random test matrix of size (l, m)
        Y (torch.Tensor): Range sketch Y = AΩ of size (m, k)
        W (torch.Tensor): Co-range sketch W = ΨA of size (l, n)
    """
    m, n = A.size()
    
    # Generate random test matrices
    Omega = torch.randn(n, k)
    Psi = torch.randn(l, m)
    
    if orthonormalize:
        Omega, _ = torch.linalg.qr(Omega)
        Psi, _ = torch.linalg.qr(Psi.T)
        Psi = Psi.T
        
    # Compute sketches
    Y = A @ Omega
    W = Psi @ A
    
    return Omega, Psi, Y, W

def low_rank_approx_svd(A, k):
    """
    Compute low-rank approximation of A using truncated SVD
    
    Args:
        A (torch.Tensor): Input matrix of size (m, n)
        k (int): Target rank
        
    Returns:
        U (torch.Tensor): Left singular vectors of size (m, k)
        S (torch.Tensor): Singular values of size (k,)
        V (torch.Tensor): Right singular vectors of size (k, n)
    """
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    U = U[:, :k]
    S = S[:k]
    V = V[:k, :]
    
    return U, S, V

def simplest_low_rank_approx(Y, W, Psi):
    """
    Implement Algorithm 3: Simplest Low-Rank Approximation
    
    Args:
        Y (torch.Tensor): Range sketch Y = AΩ of size (m, k)
        W (torch.Tensor): Co-range sketch W = ΨA of size (l, n)
        Psi (torch.Tensor): Random test matrix Psi of size (l, m)
        
    Returns:
        Q (torch.Tensor): Orthonormal basis for range of Y of size (m, q)
        X (torch.Tensor): Factor matrix of size (q, n)
        A_approx (torch.Tensor): Low-rank approximation QX of size (m, n)
    """
    m, k = Y.size()
    l, n = W.size()
    
    # Step 1: Form an orthogonal basis for the range of Y
    Q, _ = torch.linalg.qr(Y)
    q = Q.size(1)  # Rank of the approximation may be less than k
    
    # Step 2: Solve a least-squares problem to obtain X
    PsiQ = Psi @ Q
    X = torch.linalg.lstsq(PsiQ, W).solution
    
    # Step 3: Construct the rank-q approximation
    A_approx = Q @ X
    
    return Q, X, A_approx

def low_rank_approx(Y, W, Psi):
    """
    Implement Algorithm 4: Low-Rank Approximation
    
    Args:
        Y (torch.Tensor): Range sketch Y = AΩ of size (m, k)
        W (torch.Tensor): Co-range sketch W = ΨA of size (l, n)
        Psi (torch.Tensor): Random test matrix Psi of size (l, m)
        
    Returns:
        Q (torch.Tensor): Orthonormal basis for range of Y of size (m, k)
        X (torch.Tensor): Factor matrix of size (k, n)
        A_approx (torch.Tensor): Low-rank approximation QX of size (m, n)
    """
    m, k = Y.size()
    l, n = W.size()
    
    # Step 1: Form an orthogonal basis for the range of Y
    Q, _ = torch.linalg.qr(Y)
    
    # Step 2: Orthogonal-triangular factorization of ΨQ
    PsiQ = Psi @ Q
    U, T = torch.linalg.qr(PsiQ)
    
    # Step 3: Solve the least-squares problem to obtain X
    X = torch.linalg.lstsq(T, U.T @ W).solution
    
    # Step 4: Construct the rank-k approximation
    A_approx = Q @ X
    
    return Q, X, A_approx

# Generate a random matrix
m, n = 100, 80
A = torch.randn(m, n)

# Sketching approach
k, l = 20, 40
Omega, Psi, Y, W = sketch_for_low_rank_approx(A, k, l)
Q, X, A_approx1 = low_rank_approx(Y, W, Psi)

print("FUll rank grad", A.shape)
print("Project down", (Q.T@A).shape)
print("Project back", (Q@(Q.T@A)).shape)



"""Q, X, A_approx = simplest_low_rank_approx(Y, W, Psi)

Q1, X1, A_approx1 = low_rank_approx(Y, W, Psi)
# Truncated SVD approach
U, S, V = low_rank_approx_svd(A, k)

# Reconstruct low-rank approximations
A_sketch = Y @ (Psi @ Y).pinverse() @ W
A_svd = U @ torch.diag(S) @ V

# Compare approximation errors
print(f"Sketching error: {torch.norm(A - A_sketch, 'fro'):.4f}")
print(f"Truncated SVD error: {torch.norm(A - A_svd, 'fro'):.4f}")
print(f"Simplest low-rank approximation error: {torch.norm(A - A_approx, 'fro'):.4f}")
print(f"Low-rank approximation error: {torch.norm(A - A_approx1, 'fro'):.4f}")"""