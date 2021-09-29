import numpy as np
import torch

import utils.principal_curvature as curve_utils

import pytest

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class QuadraticFunction(torch.nn.Module):
    def __init__(self, diag=None, hess=None):
        super().__init__()
        
        if hess is not None:
            self.hessian = torch.tensor(hess).to(DEVICE)
        else:
            if diag is None:
                diag = [1.0, 2.0]
            self.hessian = torch.diag(torch.tensor(diag)).to(DEVICE)
        
        self.hessian = torch.nn.Parameter(self.hessian, requires_grad=False)
    
    def forward(self, x):
        return torch.dot(x, torch.matmul(self.hessian, x))


def value_grad_hess(f, point):
    value = f(point)
    grad = torch.autograd.functional.jacobian(f, point)
    hess = torch.autograd.functional.hessian(f, point)
    return value, grad, hess



@pytest.mark.parametrize('dimensions', [2, 3, 5, 10])
@pytest.mark.parametrize('radius', [0.1, 1, 2, 3, 10])
def test_sphere(dimensions, radius):
    f = QuadraticFunction(np.ones(dimensions)).to(DEVICE)
    point = torch.tensor(np.hstack((np.zeros(dimensions-1), [radius]))).to(DEVICE)
    value, pt_grad, pt_hess = value_grad_hess(f, point)
        
    iso_shape_operator, iso_curvatures, iso_directions = curve_utils.local_response_curvature_isoresponse_surface(pt_grad, pt_hess)
    iso_curvatures = iso_curvatures.detach().cpu().numpy()

    assert iso_curvatures.var() == 0

    expected_gaussian_curvature = 1 / (radius ** (dimensions - 1))
    np.testing.assert_allclose(np.abs(np.prod(iso_curvatures)), expected_gaussian_curvature)

@pytest.mark.parametrize('dimensions', [2, 3,])
@pytest.mark.parametrize('radius', [0.1, 1, 2])
def test_coordinate_transform_sphere(dimensions, radius):
    f = QuadraticFunction(np.ones(dimensions)).to(DEVICE)
    point = torch.tensor(np.hstack((np.zeros(dimensions-1), [radius]))).to(DEVICE)
    value, pt_grad, pt_hess = value_grad_hess(f, point)
        
    coordinate_transformation = -torch.eye(len(pt_grad), device=DEVICE)
    iso_shape_operator, iso_curvatures, iso_directions = curve_utils.local_response_curvature_isoresponse_surface(pt_grad, pt_hess, coordinate_transformation=coordinate_transformation)
    iso_curvatures = iso_curvatures.detach().cpu().numpy()

    assert iso_curvatures.var() == 0

    expected_gaussian_curvature = 1 / (radius ** (dimensions - 1))
    np.testing.assert_allclose(np.abs(np.prod(iso_curvatures)), expected_gaussian_curvature)