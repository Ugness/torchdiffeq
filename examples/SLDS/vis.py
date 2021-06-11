import numpy as np
from matplotlib import pyplot as plt

x = np.load('train.npy')
print(x.shape)
plt.plot(x[0,:,0],x[0,:,1])
plt.show()