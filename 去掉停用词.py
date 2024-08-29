import jieba
from collections import Counter

# 加载停用词列表
stop_words = set()
with open(r'D:\work\资源\cn_stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        stop_words.add(line.strip())

# 示例文本
text = "这是一个示例句子，展示去除停用词后的词频统计。"

# 对文本进行分词
words = jieba.cut(text)

# 去除停用词和标点符号，并统计词频
word_freq = Counter(word for word in words if word not in stop_words and word.strip())

print(word_freq)