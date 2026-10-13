"""Patch module to add missing functions to bps_torch.tools"""
import torch
import numpy as np

def sample_sphere_uniform(n_points=1024, n_dims=3, radius=1.0, random_seed=13):
    """Sample points uniformly on a sphere surface."""
    torch.manual_seed(random_seed)
    
    # Use normal distribution and normalize
    points = torch.randn(n_points, n_dims)
    points = torch.nn.functional.normalize(points, dim=1) * radius
    return points

def sample_sphere_nonuniform(n_points=1024, n_dims=3, radius=1.0, random_seed=13):
    """Sample points non-uniformly on a sphere surface."""
    torch.manual_seed(random_seed)
    
    # Similar to uniform but with different distribution
    phi = torch.rand(n_points) * 2 * torch.pi
    costheta = torch.rand(n_points) * 2 - 1
    
    x = radius * torch.sqrt(1 - costheta**2) * torch.cos(phi)
    y = radius * torch.sqrt(1 - costheta**2) * torch.sin(phi)
    z = radius * costheta
    
    return torch.stack([x, y, z], dim=1)

def sample_grid_sphere(n_points=1024, n_dims=3, radius=1.0):
    """Generate approximately n_points points on a sphere surface."""
    # Use Fibonacci lattice for approximately uniform distribution
    indices = torch.arange(0, n_points, dtype=torch.float32