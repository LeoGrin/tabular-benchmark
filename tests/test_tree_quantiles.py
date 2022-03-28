from data_transforms import tree_quantile_transformer, select_features_rf, gaussienize
from generate_data import generate_gaussian_data, import_real_data
from target_function_classif import periodic_triangle, periodic_sinus
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer

rng = np.random.RandomState(0)

# On simulation
x = generate_gaussian_data(1000, 1, regression = True, rng = rng)
y = periodic_triangle(x, period = 8, period_size=0.1, rng=rng, noise=False)

indices = np.argsort(x.reshape(-1))
plt.plot(x[indices].reshape(-1, 1), y[indices])
plt.scatter(x[indices].reshape(-1, 1), y[indices])
plt.show()

x, _, y, _ = tree_quantile_transformer(x, x, y, y, regression = True, rng = rng)
indices = np.argsort(x.reshape(-1))
plt.plot(x[indices].reshape(-1, 1), y[indices])
plt.scatter(x[indices].reshape(-1, 1), y[indices])
plt.show()

# On real data

# x, y = import_real_data(keyword = "california", max_num_samples=10000, rng=rng)
# x, y = select_features_rf(x, y, rng=rng)
# #x, y = gaussienize(x, y, type="quantile", rng=rng)
# x, y = tree_quantile_transformer(x, y, regression = False, normalize=True,  rng = rng)
#
# plt.scatter(x[:, 0], x[:, 1], c=np.array(["red", "blue"])[y], alpha=0.2, s=1)
# plt.show()
