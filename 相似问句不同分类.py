import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

start_time = time.time()
# 创建DataFrame
df = pd.read_excel('../corpus/分析_总表1026_new.xlsx')
df['sentences'] = df['问题名称']
df['label'] = df['业务标签']
# 初始化TF-IDF向量化器
tfidf_vectorizer = TfidfVectorizer()
end_time = time.time()
elapsed_time = end_time - start_time
print("初始化TF-IDF向量化器耗时：", elapsed_time, "秒")
# 计算TF-IDF矩阵
tfidf_matrix = tfidf_vectorizer.fit_transform(df['sentences'])
end_time = time.time()
elapsed_time = end_time - start_time
print("计算TF-IDF矩阵耗时：", elapsed_time, "秒")
# 计算相似度矩阵
similarity_matrix = cosine_similarity(tfidf_matrix)
end_time = time.time()
elapsed_time = end_time - start_time
print("s计算相似度矩阵耗时：", elapsed_time, "秒")
# 构建相似度矩阵的DataFrame
similarity_df = pd.DataFrame(similarity_matrix, columns=df.index, index=df.index)
end_time = time.time()
elapsed_time = end_time - start_time
print("构建相似度矩阵的DataFrame耗时：", elapsed_time, "秒")
# 获取下三角矩阵
similarity_df = similarity_df.mask(np.triu(np.ones(similarity_df.shape, dtype=np.bool_)))
similarity_df = similarity_df.fillna(0)
# 将DataFrame转换为MultiIndex Series
end_time = time.time()
elapsed_time = end_time - start_time
print("转换为MultiIndex Series耗时：", elapsed_time, "秒")
stacked_df = similarity_df.stack()
end_time = time.time()
elapsed_time = end_time - start_time
print("stack耗时：", elapsed_time, "秒")
stacked_df = stacked_df.astype(float)
# 找出大于0.8的数据
end_time = time.time()
elapsed_time = end_time - start_time
print("找出大于0.8的数据耗时：", elapsed_time, "秒")
result = stacked_df[(stacked_df > 0.7) & (stacked_df < 1)]
# 将MultiIndex Series转换回DataFrame
end_time = time.time()
elapsed_time = end_time - start_time
print("转换回DataFrame耗时：", elapsed_time, "秒")
result_df = result.reset_index()
# 重命名列名
result_df.columns = ['行', '列', '值']
# 输出结果
print(result_df)
end_time = time.time()
elapsed_time = end_time - start_time
print("耗时：", elapsed_time, "秒")
# result_df.to_excel('../corpus/分析_总表1026_new_可能是异常数据.xlsx')
# result_df = pd.read_excel('../corpus/分析_总表1026_new_可能是异常数据.xlsx')
ori_text = []
match_text = []
ori_label = []
match_label = []
score = []

for row, lie, val in result_df[['行', '列', '值']].values.tolist():
    row = int(row)
    lie = int(lie)
    val = float(val)
    if df['label'].iloc[row] != df['label'].iloc[lie]:
        print(df['sentences'].iloc[row])
        print(df['sentences'].iloc[lie])
        ori_text.append(df['sentences'].iloc[row])
        match_text.append(df['sentences'].iloc[lie])
        ori_label.append(df['label'].iloc[row])
        match_label.append(df['label'].iloc[lie])
        score.append(val)
pd.DataFrame(
    {
        '原始句子': ori_text,
        '相似句子': match_text,
        'similarity_score': score,
        '原始句子标签': ori_label,
        '相似句子标签': match_label
    }).to_excel('../corpus/分析_总表1026_new_异常数据.xlsx', index=False)
