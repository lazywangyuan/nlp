# 将大于0.5的概率转成1，其余为0
import numpy as np
arr = np.random.rand(10)
arr[arr > 0.5] = 1
arr[arr <= 0.5] = 0
print(arr)