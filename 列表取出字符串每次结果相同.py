import random

# 原始列表
chars = ['a', 'b', 'c', 'd', 'e']

# 设置不同的随机种子
random.seed(2024)

# 打乱列表顺序
random.shuffle(chars)

# 取出字符
random_char = random.sample(chars,3)
print(f"Random char: {random_char}")
