from sklearn.model_selection import ParameterSampler
#from scipy.stats.distributions import expon
import scipy.stats.distributions as distrib
import numpy as np
rng = np.random.RandomState(0)
param_grid = {'a':[1, 2], 'b': distrib.loguniform(1, 2)}
print(param_grid)
param_list = list(ParameterSampler(param_grid, n_iter=4,
                                    random_state=rng))
print(param_list)
rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
                 for d in param_list]
print(rounded_list)
print(rounded_list == [{'b': 0.89856, 'a': 1},
                  {'b': 0.923223, 'a': 1},
                  {'b': 1.878964, 'a': 2},
                  {'b': 1.038159, 'a': 2}])