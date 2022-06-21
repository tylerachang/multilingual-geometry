"""
Functions for computing distances between subspaces.
"""

import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import subspace_angles
from scipy.linalg import svd

# Random rotation matrix, rotating by some angle (in degrees) in each plane of rotation.
# See: https://en.wikipedia.org/wiki/Plane_of_rotation
# Planes of rotation selected randomly (using a uniform random orthonormal basis).
# Note: a rotation matrix is an orthonormal matrix with determinant +1. Orthonormal
# matrices have determinant +1 or -1.
def random_rotation_matrix(dim_size, angle=0):
  orthonormal_basis = ortho_group.rvs(dim_size)
  projected_rotation_matrix = np.zeros((dim_size, dim_size))
  # Fill in the projected rotation matrix.
  # 2D rotation for each pair of vectors (one rotation for each plane).
  for dim1 in range(0, dim_size, 2):
    angle_rad = np.deg2rad(angle)
    if bool(np.random.randint(0, 2)):
      angle_rad = -1.0 * angle_rad # Clockwise instead of counterclockwise.
    rotation_in_plane = np.array([[np.cos(angle_rad), -1.0 * np.sin(angle_rad)],
                                  [np.sin(angle_rad), np.cos(angle_rad)]])
    projected_rotation_matrix[dim1, dim1:dim1+2] = rotation_in_plane[0]
    projected_rotation_matrix[dim1+1, dim1:dim1+2] = rotation_in_plane[1]
  # Fill in the last dimension for odd dim_size.
  if dim_size % 2 != 0:
    projected_rotation_matrix[-1, -1] = 1.0
  # Rotation matrix: change to random orthonormal basis, rotate, change back to original basis.
  rotation_matrix = np.matmul(np.matmul(orthonormal_basis, projected_rotation_matrix), orthonormal_basis.T)
  return rotation_matrix, orthonormal_basis

# Scale by some multiplier in each dimension.
def random_scaling(dim_size, scale=1.0):
    scaling = np.zeros(dim_size)
    random_bools = np.random.choice([True, False], size=dim_size, replace=True)
    scaling[random_bools] = scale
    scaling[np.logical_not(random_bools)] = 1.0 / scale
    return scaling


# Computes the distance between two subspaces with shapes (dim_size, dim_a) and (dim_size, dim_b).
# Optionally can include the remaining dimensions (subspace shapes: dim_size, dim_size) and
# specify the subspace size. Some metrics (e.g. Riemann) also require
# the s vectors (s_a, s_b) from SVD with shapes (dim_a) and (dim_b). When squared,
# these should be proportional (scaled up by n_tokens - 1) to the variance in each dimension.
# If subspaces were computed from different n_tokens, then the s vectors should be scaled down
# by \sqrt(n_tokens-1) for each language.
def principal_angles_and_vectors(subspace_a, subspace_b):
  # Computes the principal angles (in decreasing order) and corresponding principal vectors.
  # https://www.intel.com/content/www/us/en/develop/documentation/mkl-cookbook/top/computing-principal-angles-between-two-subspaces.html
  utw = np.matmul(np.transpose(subspace_a), subspace_b) # dim_a x dim_b
  # k = min(dim_a, dim_b)
  # Shapes: (dim_a, k), (k), (k, dim_b).
  u, s_utw, vh = svd(utw, full_matrices=False, compute_uv=True, overwrite_a=True)
  v = np.transpose(vh) # dim_b x k
  s_utw = np.clip(s_utw, -1.0, 1.0) # Account for numerical issues leading to values outside [-1, 1].
  # SVs are non-negative and in decreasing order, so angles in [0, \pi/2] are in increasing order.
  principal_angles = np.flip(np.arccos(s_utw)) # Angles in decreasing order. Shape k.
  principal_angles = np.real_if_close(principal_angles, tol=1000) # Tolerance in machine epsilons.
  u = np.flip(u, axis=-1) # Flip corresponding vectors in U and V.
  v = np.flip(v, axis=-1)
  # Principal vectors.
  a_vectors = np.matmul(subspace_a, u) # dim_size x k
  b_vectors = np.matmul(subspace_b, v) # dim_size x k
  return (principal_angles, a_vectors, b_vectors)
def subspace_distance(subspace_a, subspace_b, metric="mean_principal_angle",
                      s=None, dim_a=None, dim_b=None):
  if dim_a is None:
    dim_a = subspace_a.shape[-1]
  if dim_b is None:
    dim_b = subspace_b.shape[-1]
  assert subspace_a.shape[0] == subspace_b.shape[0], "Unequal dimension of larger space."
  dim_size = subspace_a.shape[0]
  # Restrict subspaces to the correct sizes.
  subspace_a = subspace_a[:, :dim_a]
  subspace_b = subspace_b[:, :dim_b]
  # Compute distance metric.
  if metric == "mean_principal_angle": # Returns in radians.
    # Inputs: (dim_size, subspace_size).
    # Output: min(dim_a, dim_b) principal angles.
    principal_angles = subspace_angles(subspace_a, subspace_b)
    return np.mean(principal_angles)
  elif metric == "geodesic_distance":
    # Semantic spaces: Measuring the distance between different subspaces (Zuccon et al.).
    principal_angles = subspace_angles(subspace_a, subspace_b)
    return np.sqrt(np.sum(np.square(principal_angles)))
  elif metric == "chordal_distance":
    # Semantic spaces: Measuring the distance between different subspaces (Zuccon et al.).
    projection_a = np.matmul(subspace_a, np.transpose(subspace_a)) # dim_size x dim_size
    projection_b = np.matmul(subspace_b, np.transpose(subspace_b)) # dim_size x dim_size
    trace = np.trace(np.matmul(projection_a, projection_b))
    chordal_distance = np.sqrt(max(dim_a, dim_b) - trace)
    return chordal_distance
  elif metric == "riemann_full_distance":
    # S^2 and Vh from SVD define an ellipsoid, where axes have magnitudes proportional
    # to variance in that direction (scaled up by n_tokens - 1). This ellipsoid can
    # also be defined by the positive semi-definite matrix that scales by S^2 in those
    # directions (eigenvalues S^2 and eigenvectors rows from Vh).
    # Requires the entire subspaces (e.g. subspace_a has shape (dim_size, dim_size)).
    # Riemannian metric and geometric mean for positive semidefinite matrices of
    # fixed rank (Bonnabel & Sepulchre, 2009).
    # Geometric distance between positive definite matrices of different
    # dimensions (Lim et al., 2019).
    assert (dim_a == dim_size and dim_b == dim_size), "Subspaces must span the entire space."
    # Get the corresponding linear transformations for each subspace, in the standard
    # basis (change to SVD basis, scale by S^2, change back to standard basis).
    # These should be positive semi-definite. These matrices are equal to X^T X, where
    # X is the corresponding data matrix. If the data is zero-centered, then this
    # is equal to the covariance matrix multiplied by num_tokens-1.
    a_inv = np.transpose(subspace_a) # np.linalg.inv(subspace_a) # Because subspace_a is orthonormal.
    matrix_a = np.matmul(subspace_a, np.matmul(np.diag(np.square(s[0])), a_inv))
    b_inv = np.transpose(subspace_b) # np.linalg.inv(subspace_b) # Because subspace_b is orthonormal.
    matrix_b = np.matmul(subspace_b, np.matmul(np.diag(np.square(s[1])), b_inv))
    lambdas = np.linalg.eigvals(np.matmul(np.linalg.inv(matrix_a), matrix_b))
    # Machine epsilon is usually roughly 2.22e-16.
    # lambdas = np.real_if_close(lambdas, tol=1000) # Tolerance in machine epsilons.
    lambdas = np.real(lambdas) # Eigenvalues should be real except for numerical errors.
    # Eigenvalues should be positive except for numerical errors, so ignore NaN log values.
    transformed_lambdas = np.square(np.log(lambdas))
    riemann_distance = np.sqrt(np.nansum(transformed_lambdas))
    return riemann_distance
  elif metric == "riemann_distance":
    # Similar to metric above, but extended to fixed rank k = dim_a = dim_b < dim_size.
    # Not a true distance metric. Metric obtained from (Equation 15):
    # Riemannian metric and geometric mean for positive semidefinite matrices of
    # fixed rank (Bonnabel & Sepulchre, 2009).
    assert dim_a == dim_b, "Subspaces must have equal rank."
    # Constant to weight distance within a subspace (distance on the cone) vs.
    # distance between the subspaces in which the ellipsoids are contained.
    # Set to one so that the distance is equal to the usual Riemann distance
    # when the two subspace spans are equal (e.g. when k = dim_size). However,
    # the implementation above is more efficient in these cases.
    weight = 1.0
    # In this case, we want to use the transpose instead of the inverse.
    a_transp = np.transpose(subspace_a)
    matrix_a = np.matmul(subspace_a, np.matmul(np.diag(np.square(s[0][:dim_a])), a_transp))
    b_transp = np.transpose(subspace_b)
    matrix_b = np.matmul(subspace_b, np.matmul(np.diag(np.square(s[1][:dim_b])), b_transp))
    # Get principal angles and vectors.
    principal_angles, a_vectors, b_vectors = principal_angles_and_vectors(subspace_a, subspace_b)
    r2_a = np.matmul(np.transpose(a_vectors), np.matmul(matrix_a, a_vectors)) # k x k
    r2_b = np.matmul(np.transpose(b_vectors), np.matmul(matrix_b, b_vectors)) # k x k
    lambdas = np.linalg.eigvals(np.matmul(r2_a, np.linalg.inv(r2_b)))
    lambdas = np.real(lambdas) # Eigenvalues should be real except for numerical errors.
    term2 = np.nansum(np.square(np.log(lambdas)))
    term1 = np.sum(np.square(principal_angles))
    riemann_distance = np.sqrt(term1 + weight * term2)
    return riemann_distance
  print("Unknown subspace distance metric.")
  return np.NaN
