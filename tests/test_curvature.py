import numpy as np
from scipy.stats import ortho_group
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


def hyperboloid_graph(x_vals, y_vals, a, c):
    z_sq = c**2 * (x_vals**2 / a**2 + y_vals**2 / a**2 - 1)
    if type(x_vals) == type(torch.Tensor()):
        return torch.sqrt_(z_sq)
    else:
        return np.sqrt(z_sq)


def hyperboloid_eq(x, a, c):
    return (x[:-1] ** 2) / (a ** 2) - x[-1] ** 2 / c ** 2


def value_grad_hess(f, point):
    value = f(point)
    grad = torch.autograd.functional.jacobian(f, point)
    hess = torch.autograd.functional.hessian(f, point)
    return value, grad, hess


def test_hyperboloid():
    a = 1
    c = 3
    f = lambda x: hyperboloid_graph(x[0], x[1], a=a, c=c)
    x = 2.0
    y = 2.0
    point = torch.tensor([x, y]).to(DEVICE)
    z, grad, hess = value_grad_hess(f, point)

    shape_operator, principal_curvatures, principal_directions = curve_utils.local_response_curvature_graph(grad, hess)

    u = z.cpu().numpy() / c
    analytic_gauss_curvature = -c**2 / (c**2 + (a**2 + c**2) * u**2)**2

    np.testing.assert_allclose(
        analytic_gauss_curvature,
        torch.det(shape_operator).cpu().numpy(),
        rtol=1e-6
    )


@pytest.mark.parametrize('dimensions', [2, 3, 5, 10])
@pytest.mark.parametrize('radius', [0.1, 1, 2, 3, 10])
def test_sphere(dimensions, radius):
    f = QuadraticFunction(np.ones(dimensions)).to(DEVICE)
    #point = torch.tensor(np.hstack((np.zeros(dimensions-1), [radius]))).to(DEVICE)
    point = torch.tensor(np.ones(dimensions) / np.sqrt(dimensions) * radius).to(DEVICE)
    value, pt_grad, pt_hess = value_grad_hess(f, point)

    iso_shape_operator, iso_curvatures, iso_directions = curve_utils.local_response_curvature_isoresponse_surface(pt_grad, pt_hess)
    iso_curvatures = iso_curvatures.detach().cpu().numpy()

    assert iso_curvatures.var() < 1e-16

    expected_gaussian_curvature = 1 / (radius ** (dimensions - 1))
    np.testing.assert_allclose(np.abs(np.prod(iso_curvatures)), expected_gaussian_curvature)


@pytest.mark.parametrize('dimensions', [2, 3, 5])
@pytest.mark.parametrize('radius', [0.1, 1, 2])
def test_coordinate_transform_sphere(dimensions, radius):
    f = QuadraticFunction(np.ones(dimensions)).to(DEVICE)
    #point = torch.tensor(np.hstack((np.zeros(dimensions-1), [radius]))).to(DEVICE)
    point = torch.tensor(np.ones(dimensions) / np.sqrt(dimensions) * radius).to(DEVICE)
    
    value, pt_grad, pt_hess = value_grad_hess(f, point)

    #coordinate_transformation = -torch.eye(len(pt_grad), device=DEVICE)
    coordinate_transformation = torch.tensor(ortho_group.rvs(len(point)), dtype=torch.double).to(DEVICE)

    iso_shape_operator, iso_curvatures, iso_directions = curve_utils.local_response_curvature_isoresponse_surface(pt_grad, pt_hess)

    iso_curvatures = iso_curvatures.detach().cpu().numpy()

    assert iso_curvatures.var() < 1e-16

    expected_gaussian_curvature = 1 / (radius ** (dimensions - 1))
    np.testing.assert_allclose(np.abs(np.prod(iso_curvatures)), expected_gaussian_curvature)


@pytest.mark.skip("Will usually fail because the transformation reults in bad implicit functions")
def test_coordinate_transform():
    f = QuadraticFunction(np.array([1.0, 2.0, 0.1, 0.5])).to(DEVICE)
    point = torch.tensor(np.hstack((np.zeros(3), [1]))).to(DEVICE)

    value, pt_grad, pt_hess = value_grad_hess(f, point)

    iso_shape_operator1, iso_curvatures1, iso_directions1 = curve_utils.local_response_curvature_isoresponse_surface(pt_grad, pt_hess)
    iso_curvatures1 = iso_curvatures1.detach().cpu().numpy()

    M = ortho_group.rvs(len(point))
    #M = np.eye(len(point))
    #_M = ortho_group.rvs(len(point) - 1)
    #M[:-1, :-1] = _M
    coordinate_transformation = torch.tensor(M, dtype=torch.double).to(DEVICE)
    iso_shape_operator2, iso_curvatures2, iso_directions2 = curve_utils.local_response_curvature_isoresponse_surface(pt_grad, pt_hess, coordinate_transformation=coordinate_transformation)
    iso_curvatures2 = iso_curvatures2.detach().cpu().numpy()

    np.testing.assert_allclose(iso_curvatures1, iso_curvatures2)


@pytest.mark.parametrize('dimensions', [20, 40, 60, 500])
@pytest.mark.parametrize('radius', [0.1, 1, 2])
def test_bad_condition(dimensions, radius):
    f_3d = QuadraticFunction([3.0, 3.0, 1.0]+[0.5] * (dimensions - 3)).to(DEVICE)
    point = torch.tensor(
        [np.sqrt(0.5), np.sqrt(0.5), 0]+[0] * (dimensions - 3),
        dtype=torch.float
    ).to(DEVICE) * radius

    # first in original coordinates
    value_orig, pt_grad_orig, pt_hess_orig = value_grad_hess(f_3d, point)
    with pytest.raises(ValueError):
        iso_shape_operator_orig, iso_curvatures_orig, iso_directions_orig = \
            curve_utils.local_response_curvature_isoresponse_surface(
                pt_grad_orig,
                pt_hess_orig,
                coordinate_transformation=torch.eye(len(point)).to(DEVICE)
            )

    rst = None
    # This will in general result in very bad conditioning
    M = ortho_group.rvs(len(point), random_state=rst)
    coordinate_transformation = torch.tensor(M).to(DEVICE)

    def new_f(x):
        return f_3d(torch.matmul(coordinate_transformation.T.type(x.dtype), x))
    new_point = torch.matmul(
        coordinate_transformation.type(point.dtype),
        point
    ).detach().clone()

    value_new, pt_grad_new, pt_hess_new = value_grad_hess(new_f, new_point)

    np.testing.assert_allclose(
        value_orig.cpu().numpy(),
        value_new.cpu().numpy(),
        rtol=1e-6
    )

    iso_shape_operator_new, iso_curvatures_new, iso_directions_new = \
        curve_utils.local_response_curvature_isoresponse_surface(
            pt_grad_new, pt_hess_new,
        )

    expected_curvatures = np.array(
        [-1/6] * (dimensions - 3) + [-1/3, -1.0]
    ) / radius

    np.testing.assert_allclose(
        iso_curvatures_new.cpu().numpy(),
        expected_curvatures,
        rtol=1e-6,
    )