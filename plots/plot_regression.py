import numpy as np
import pickle
import matplotlib.pyplot as plt

# x_train, x_test, y_train, y_test = pickle.load(open('saved_models/regression/california/data', 'rb'))
#
# print(x_train.shape)
# print(y_train.shape)
#
# rf = pickle.load(open('saved_models/regression/california/rf', 'rb'))
# mlp = pickle.load(open('saved_models/regression/california/mlp', 'rb'))
#
#
# plt.scatter(x_train, y_train, color='blue', s=0.5, label="data", alpha=0.4)
#
# x_grid = np.linspace(np.min(x_train), np.max(x_train), 100).astype(np.float32)
# y_rf = rf.predict(x_grid.reshape(-1, 1))
# y_mlp = mlp.predict(x_grid.reshape(-1, 1))
#
# plt.plot(x_grid, y_rf, color='red', label="rf")
#
# plt.plot(x_grid, y_mlp, color='green', label="mlp")
#
# plt.legend()
#
# plt.show()
#
#x = np.linspace(-3, 3, 100)
# y = periodic(x, offset = 0.8, period = 3, noise=False)
# print(x)
# print(y)
# plt.plot(x, y)
# plt.show()

# offsets = [0]
# periods = [16]
# #period_sizes = [0.1, 0.2, 0.3]
# num_samples_list = [500]
#
# offset = 0
#
# for num_samples in num_samples_list:
#     for period in periods:
#         for iter in range(2):

filename_rf = "../saved_models/regression_synthetic_pretrained/pretrained/-1328826748526267170.pkl"
#filename_mlp = "saved_models/regression_synthetic/mlp/-6905694080875447095"
filename_mlp = "saved_models/regression_synthetic_pretrained/pretrained/-661633476578852768"

filename_mlp_2 = "saved_models/regression_synthetic_pretrained/pretrained/7702155688162513620"

x_train, x_test, y_train, y_test = pickle.load(open(filename_mlp + ".data", 'rb'))
plt.scatter(x_train, y_train, color='blue', s=2, label="data", alpha=1)
print(x_train.shape)
print(y_train.shape)

#rf = pickle.load(open('saved_models/regression_synthetic_{}_{}_{}_{}_rf'.format(num_samples, offset, period, iter), 'rb'))
with open(filename_rf, 'rb') as f:
    rf = pickle.load(f)
print(rf)
#mlp = pickle.load(open('saved_models/regression_synthetic/{}/mlp'.format(period), 'rb'))
with open(filename_mlp + ".pkl", 'rb') as f:
    mlp = pickle.load(f)
#mlp = pickle.load(open('saved_models/regression_synthetic_{}_{}_{}_{}_mlp_pickle.pkl'.format(num_samples, offset, period, iter), 'rb'))
print(mlp)

with open(filename_mlp_2 + ".pkl", 'rb') as f:
    mlp_2 = pickle.load(f)
#mlp = pickle.load(open('saved_models/regression_synthetic_{}_{}_{}_{}_mlp_pickle.pkl'.format(num_samples, offset, period, iter), 'rb'))


x_grid = np.linspace(np.min(x_train), np.max(x_train), 100).astype(np.float32)
y_rf = rf.predict(x_grid.reshape(-1, 1))
y_mlp = mlp.predict(x_grid.reshape(-1, 1))
y_mlp_2 = mlp_2.predict(x_grid.reshape(-1, 1))
plt.plot(x_grid, y_rf, color='green', label="noise=0", alpha=1)
plt.plot(x_grid, y_mlp, color='blue', label="noise=0.01", alpha=1)

plt.plot(x_grid, y_mlp_2, color='black', label="noise=0.05", alpha=1)

#plt.title("num samples: {}, offset: {}, period: {}".format(num_samples, offset, period))

plt.ylim([-2, 4])

plt.xlim([-2, 2])

plt.legend()

plt.show()