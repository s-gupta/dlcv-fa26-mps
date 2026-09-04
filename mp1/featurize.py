import numpy as np

def rectified_random_projection(x, num_rrp, seed, max_features=8192):
  assert num_rrp <= max_features, \
      f'num_rrp {num_rrp} exceeds the {max_features} sampled projections'
  rng_sigma = np.random.RandomState(seed)
  rng_bias = np.random.RandomState(seed + 1)
  sigma = rng_sigma.normal(size=(x.shape[1], max_features)) / np.sqrt(x.shape[1])
  bias = rng_bias.normal(size=max_features) * 0.1
  return np.maximum(x @ sigma[:, :num_rrp] + bias[:num_rrp], 0.0)

def featurize(x, type='raw', num_rrp=None, seed=None):
    if type == 'raw':
        x = x.reshape(x.shape[0], -1) - 0.5
    elif type == 'rrp':
        x = x.reshape(x.shape[0], -1) - 0.5
        x = rectified_random_projection(x, num_rrp, seed)
    return x
