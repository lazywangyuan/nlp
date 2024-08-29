import numpy as np
import pandas as pd


def len_text_min_max(df, n, max_min_n):
    select_list = []
    # 创建示例数据
    # 计算文本长度并添加到DataFrame
    df['length'] = df['文本'].apply(len)
    # 假设有一组文本长度数据
    text_lengths = df['length'].values.tolist()
    # 计算平均值和标准差
    mean_length = np.mean(text_lengths)
    std_dev = np.std(text_lengths)
    # 选择符合正态分布的数据
    x = 1  # 标准差倍数
    selected_data = [length for length in text_lengths if
                     mean_length - x * std_dev <= length <= mean_length + x * std_dev]
    # 从符合条件的数据中选择n条
    n = n - (max_min_n * 2)
    # 计算中间位置的索引
    selected_data.sort()
    middle_index = len(selected_data) // 2
    # 使用切片操作从中间位置开始取n条数据
    start_index = middle_index - n // 2
    end_index = start_index + n
    selected_data = selected_data[start_index:end_index]
    select_list.extend(selected_data)
    # 选择5条最短和5条最长的数据
    text_lengths.sort()
    shortest_data = text_lengths[:max_min_n]
    longest_data = text_lengths[-max_min_n:]
    select_list.extend(shortest_data)
    select_list.extend(longest_data)
    # 并且已经初始化了 select_list
    select_df_list = []
    for sel_num in select_list:
        selected_data = df.loc[df['length'] == sel_num].sample(n=1)
        select_df_list.append(selected_data)
    # 最后，将 select_list 中的所有结果合并成一个新的DataFrame
    combined_data = pd.concat(select_df_list)
    combined_data.to_csv('../corpus/正太分布后的数据.csv', index=False)
    print(combined_data)
    print(len(combined_data))
