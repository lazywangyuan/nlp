# 测试在cpu、gpu上的表现
"""
    1分钟能跑多个个样本呢
    1个样本需要多少时间
    模型大小
"""
import time


# 模拟处理一个样本的函数
def process_sample(sample):
    # 这里可以填入你的样本处理代码
    pass


# 模拟获取样本的函数
def get_sample():
    # 这里可以填入你的样本获取代码
    pass


# 测试一分钟内处理多少个样本
def test_throughput():
    start_time = time.time()
    samples_processed = 0
    while time.time() - start_time < 60:
        sample = get_sample()
        process_sample(sample)
        samples_processed += 1

    print("在一分钟内处理了 %d 个样本。" % samples_processed)


if __name__ == "__main__":
    test_throughput()
