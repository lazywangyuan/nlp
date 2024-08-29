import tensorflow as tf
import numpy as np
import os

# tensorflow-gpu==1.15.4
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
config = tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)
# 90000000 4327
# 60000000 2279
# 40000000 1255
# 30000000 1255
# 20000000 743
# 9000000  487 500MB
# 使用numpy生成100个随机点
def count_g(num):
    random_num = 0
    if num == 0.5:
        random_num = 9000000
    elif num == 1:
        random_num = 30000000
    elif num == 2:
        random_num = 60000000
    elif num == 4:
        random_num = 90000000
    else:
        print('在500M ,1G,2G,4G中选择')
    return random_num


# x_data = np.random.rand(900000000)
x_data = np.random.rand(count_g(1))
y_data = x_data * 0.1 + 0.2

# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)
# # 初始化变量
init = tf.global_variables_initializer()

with session as sess:
    sess.run(init)
    for step in range(20100000):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k, b]))
