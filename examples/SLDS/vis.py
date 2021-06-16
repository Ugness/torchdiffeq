import numpy as np
from matplotlib import pyplot as plt

x = np.load('train.npy')
print(x.shape)
plt.scatter(x[:, :, 0],x[:, :, 1])
plt.savefig('data.png')
plt.show()
