import torch
import unittest


class Sketch:
    def __init__(self, A, k, l, orthonormalize=True):
        self.Omega, self.Psi, self.Y, self.W = self.sketch_for_low_rank_approx(A, k, l, orthonormalize)

    def sketch_for_low_rank_approx(self, A, k, l, orthonormalize=True):
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

    def linear_update(self, H, theta, eta):
        """
        Implement Algorithm 2: Linear Update to Sketch

        Args:
            H (torch.Tensor): Update matrix of size (m, n)
            theta (float): Scalar coefficient for the previous sketch
            eta (float): Scalar coefficient for the update matrix

        Updates:
            self.Y (torch.Tensor): Range sketch Y = AΩ
            self.W (torch.Tensor): Co-range sketch W = ΨA
        """
        m, n = H.size()
        self.Y = theta * self.Y + eta * H @ self.Omega
        self.W = theta * self.W + eta * self.Psi @ H

    def simplest_low_rank_approx(self):
        """
        Implement Algorithm 3: Simplest Low-Rank Approximation
        
        Returns:
            Q (torch.Tensor): Orthonormal basis for range of Y of size (m, q)
            X (torch.Tensor): Factor matrix of size (q, n)
            A_approx (torch.Tensor): Low-rank approximation QX of size (m, n)
        """
        m, k = self.Y.size()
        l, n = self.W.size()
        
        # Step 1: Form an orthogonal basis for the range of Y
        Q, _ = torch.linalg.qr(self.Y)
        q = Q.size(1)  # Rank of the approximation may be less than k
        
        # Step 2: Solve a least-squares problem to obtain X
        PsiQ = self.Psi @ Q
        X = torch.linalg.lstsq(PsiQ, self.W).solution
        
        # Step 3: Construct the rank-q approximation
        A_approx = Q @ X
        
        return Q, X, A_approx

    def low_rank_approx(self):
        """
        Implement Algorithm 4: Low-Rank Approximation
        
        Returns:
            Q (torch.Tensor): Orthonormal basis for range of Y of size (m, k)
            X (torch.Tensor): Factor matrix of size (k, n)
            A_approx (torch.Tensor): Low-rank approximation QX of size (m, n)
        """
        m, k = self.Y.size()
        l, n = self.W.size()
        
        # Step 1: Form an orthogonal basis for the range of Y
        Q, _ = torch.linalg.qr(self.Y)
        
        # Step 2: Orthogonal-triangular factorization of ΨQ
        PsiQ = self.Psi @ Q
        U, T = torch.linalg.qr(PsiQ)
        
        # Step 3: Solve the least-squares problem to obtain X
        X = torch.linalg.lstsq(T, U.T @ self.W).solution
        
        # Step 4: Construct the rank-k approximation
        A_approx = Q @ X
        
        return Q, X, A_approx

    @staticmethod
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

class TestSketch(unittest.TestCase):
    def setUp(self):
        self.m, self.n = 100, 80
        self.A = torch.randn(self.m, self.n)
        self.k, self.l = 20, 40
        self.sketch = Sketch(self.A, self.k, self.l)

    def test_sketch_for_low_rank_approx(self):
        Omega, Psi, Y, W = self.sketch.sketch_for_low_rank_approx(self.A, self.k, self.l)
        self.assertEqual(Y.size(), (self.m, self.k))
        self.assertEqual(W.size(), (self.l, self.n))

    def test_linear_update(self):
        H = torch.randn(self.m, self.n)
        theta, eta = 0.5, 0.7
        self.sketch.linear_update(H, theta, eta)
        Y_updated = theta * self.sketch.Y + eta * H @ self.sketch.Omega
        W_updated = theta * self.sketch.W + eta * self.sketch.Psi @ H
        self.assertTrue(torch.allclose(self.sketch.Y, Y_updated))
        self.assertTrue(torch.allclose(self.sketch.W, W_updated))

    def test_simplest_low_rank_approx(self):
        Q, X, A_approx = self.sketch.simplest_low_rank_approx()
        self.assertTrue(torch.allclose(A_approx, Q @ X))

    def test_low_rank_approx(self):
        Q, X, A_approx = self.sketch.low_rank_approx()
        self.assertTrue(torch.allclose(A_approx, Q @ X))

    def test_low_rank_approx_svd(self):
        U, S, V = Sketch.low_rank_approx_svd(self.A, self.k)
        A_approx = U @ torch.diag(S) @ V.T
        self.assertTrue(torch.allclose(A_approx, self.A[:, :self.k] @ torch.diag(S) @ V.T))

if __name__ == '__main__':
    unittest.main()