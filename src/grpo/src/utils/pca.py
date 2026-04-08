import torch
import torch.nn as nn

class DifferentiablePCA(nn.Module):
    """
    Differentiable PCA transformer.
    Usage:
      pca = DifferentiablePCA(n_components=K, device='cuda')
      pca.fit(X_train)                # X_train: [N, D] torch tensor (float)
      Z = pca.transform(X)            # X: [B, D], returns [B, K], gradient flows to X
      Xrec = pca.inverse_transform(Z) # reconstruction [B, D]

    Notes:
    - fit() performs SVD on centered data and stores mean & components (not updated by default).
    - transform() is linear: (X - mean) @ components.T -> differentiable wrt X.
    - For large D (e.g., L*C huge) use subsampling or apply pooling before fit.
    """

    def __init__(self, n_components: int, eps: float = 1e-8, device: str = None):
        super().__init__()
        self.n_components = int(n_components)
        self.eps = eps
        self.register_buffer('_mean', torch.zeros(1))      # will be replaced on fit
        self.register_buffer('_components', torch.zeros(1))# shape [K, D]
        self.fitted = False
        if device is not None:
            self.to(device)

    def fit(self, X: torch.Tensor, center: bool = True, svd_solver: str = 'svd'):
        """
        Fit PCA components from data X.
        X: [N, D] torch tensor (float)
        center: whether to subtract mean
        svd_solver: 'svd' uses torch.linalg.svd
        """
        assert X.dim() == 2, "X must be 2D [N, D]"
        device = X.device
        N, D = X.shape
        if self.n_components > D:
            raise ValueError("n_components must <= D")
        # compute mean
        mean = X.mean(dim=0, keepdim=True) if center else torch.zeros(1, D, device=device)
        Xc = X - mean if center else X

        # compute SVD on Xc (N x D). For efficiency, if N < D, use Xc @ Xc.T trick maybe.
        # Using full SVD here (torch.linalg.svd). Might be slow for huge dims.
        # We compute V (right singular vectors) and take first K rows as components.
        # Xc = U S V^T  -> V^T shape [D, D], so V[:K] are top K components
        # Use economic SVD
        try:
            # compute compact SVD; returns U, S, Vh where Vh shape [D, D]
            # torch.linalg.svd returns U, S, Vh; Vh is conjugate transpose of V
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        except Exception as e:
            # fallback: compute covariance and eig
            cov = (Xc.T @ Xc) / max(N - 1, 1)
            Svals, V = torch.linalg.eigh(cov)  # ascending order
            # take descending
            idx = torch.argsort(Svals, descending=True)
            V = V[:, idx]
            Vh = V.T
        # Vh: [D, D] ; take top n_components rows
        components = Vh[:self.n_components, :]   # shape [K, D]
        # store
        self._mean = mean.detach()
        self._components = components.detach()
        self.fitted = True
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project X to PCA coords.
        X: [B, D] -> returns [B, K]
        gradient flows to X (and to components if components requires_grad=True)
        """
        assert X.dim() == 2, "X must be 2D [B, D]"
        if not self.fitted:
            raise RuntimeError("PCA not fitted. Call fit() first.")
        # broadcast mean
        mean = self._mean
        comps = self._components  # [K, D]
        Xc = X - mean
        Z = Xc @ comps.T          # [B, K]
        return Z

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct X from Z.
        Z: [B, K] -> returns [B, D]
        """
        assert Z.dim() == 2
        comps = self._components   # [K, D]
        Xrec = Z @ comps          # [B, D] since comps is [K, D]
        # add mean
        Xrec = Xrec + self._mean
        return Xrec

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # ensure buffers in correct device/dtype
        return self
