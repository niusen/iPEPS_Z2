"""Exact thin SVD, same CPU/CUDA driver convention as the checked CTMRG path."""
import torch


def full_svd(a):
    return torch.linalg.svd(a,full_matrices=False,**({'driver':'gesvd'} if a.is_cuda else {}))
