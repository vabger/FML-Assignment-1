import numpy as np

def initialise_input(N, d):
  '''
  N: Number of vectors
  d: dimension of vectors
  '''
  np.random.seed(0)
  U = np.random.randn(N,d)
  M1 = np.abs(np.random.randn(d, d))
  M2 = np.abs(np.random.randn(d, d))


  return U, M1, M2

def solve(N, d):
  U, M1, M2 = initialise_input(N, d)

  '''
  Enter your code here for steps 1 to 6
  '''

  # Step 1
  X = U @ M1
  Y = U @ M2

  # Step 2
  offsets = np.arange(1,N+1).reshape(N,1)
  X_hat = X + offsets

  # Step 3
  Z = X_hat @ Y.T
  row_ind = np.reshape(np.arange(1,N+1),(-1,1))
  col_ind = np.arange(1,N+1)

  mask = (row_ind + col_ind - 1) % 2

  Z = Z * mask


  # Step 4
  Z_exp = np.exp(Z)
  row_sum = np.sum(Z_exp,axis=1).reshape(-1,1)
  Z_hat = Z_exp/row_sum
  

  # Step 5
  max_indices = np.argmax(Z_hat, axis=1)


  return max_indices
  
