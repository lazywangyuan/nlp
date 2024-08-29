import jieba
from collections import defaultdict

# 假设我们有以下文章列表
articles = [
    "这是一篇关于机器学习的文章。",
    "这是一篇关于人工智能的文章。",
    "这是一篇关于大数据的文章。",
    "这是另一篇关于机器学习的文章。",
]

# 需要统计的句子
sentence = "机器学习"
top = 2
# 对文章进行分词
words_list = []
for article in articles:
    words = jieba.lcut(article)
    words_list.append(words)

# 统计每个字在所有文章中出现的次数
word_counts = defaultdict(int)
for words in words_list:
    for word in words:
        word_counts[word] += 1

print()
que_cut = jieba.lcut(sentence)
print(words_list)
num_list = []
for all_list in words_list:
    num_list.append(len(list(set(que_cut) - set(all_list))))

print(num_list)
# 排序列表
# 原始列表
my_list = [0, 2, 2, 0]
# 使用enumerate获取元素和索引，然后根据元素值排序
sorted_items = sorted(enumerate(my_list), key=lambda item: item[1], reverse=True)
# 打印排序后的元素和它们原来的索引
cal_list = []
for index, value in sorted_items:
    cal_list.append(articles[index])
    print(f'Original text:{articles[index]}')
print(cal_list[:top])
