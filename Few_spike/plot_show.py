import math
import numpy as np
import torch
from matplotlib import pyplot as plt

from fs_coding import fs_relu

x = np.linspace(-5, 5, 1000, dtype=np.float32)
x = torch.Tensor(x)
# x = torch.tensor(0.42, dtype=torch.float32)

# # y_swish = fs_swish(x)


# ttt = nn.ReLU()
# y_relu = ttt(x)
y_fs_relu = fs_relu(x)

# print(y_fs_relu)
# plt.plot(x, y_relu)
plt.plot(x, y_fs_relu)
plt.show()
