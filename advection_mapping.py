import numpy as np
import scipy.integrate

def advection_map(x:np.array, u:np.array, source_time_functions:tuple):
  ''' Maps (x, u) to values of each source time function provided at
  the corresponding position x. '''
  # Rearrange input (x,u) into linear arrays
  isort = np.argsort(x.ravel())
  x_lin = x.ravel()[isort]
  u_lin = u.ravel()[isort]
  # Numerically integrate slowness
  taus = scipy.integrate.cumulative_trapezoid(1/u_lin, x=x, initial=0)
  # Set up numerical coordinate map from t -> x given velocity field
  coord_map = lambda xq: np.interp(xq, x_lin, taus)
  # Map x -> t -> source quantity
  return tuple(f(-coord_map(x)) for f in source_time_functions)