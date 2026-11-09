import numpy as np

def svd(A):
  m,n = A.shape

  AT = A.T

  B = AT @ A
  
  eigenvalues , eigenvectors = np.linalg.eigh(B)

  idx = eigenvalues.argsort()[::-1]
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:,idx]

  eigenvalues = np.maximum(eigenvalues,0)

  singular_values = np.sqrt(eigenvalues)

  tol = np.finfo(float).eps*max(m,n)*singular_values[0]

  mask = singular_values > tol
  singular_values = singular_values[mask]

  V = eigenvectors[:,mask]

  U = (A @ V) / singular_values[None, :]

  sigma = np.zeros((m, n))
  k = len(singular_values)
  sigma[:k, :k] = np.diag(singular_values)

  return U, sigma, V.T

def truncated_svd(A, k):
  U, sigma, V = svd(A)
  return U[:,:k], sigma[:k,:k], V[:k,:]