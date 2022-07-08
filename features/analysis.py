import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("shopping", "rb") as f:
    counts = np.array(pickle.load(f))

plt.bar(range(counts.shape[1]), np.max(counts, axis=0))
plt.bar(range(counts.shape[1]), np.min(counts, axis=0))
plt.show()
