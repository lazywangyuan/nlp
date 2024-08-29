# 常见类别：公积金、居住证、参保、企业
# 不常见类别:公共服务、海关口岸、文体教育、入伍
# 从历史数据中提取对应类别，之后做正太分布提取
import pandas as  pd
import numpy as np
# from hkGoogleTrans import translated_content
from tqdm import tqdm
import jieba.posseg as pseg
# import baidu_translate as fanyi
import time

# 使用列表推导式来筛选包含关键字的列表
keywords = ["公积金", "居住证", "参保", "企业", "公共服务", "海关", "教育", "入伍", "军队", "参军"]


def len_text_min_max(df, n):
    select_list = []
    # 创建示例数据
    # 计算文本长度并添加到DataFrame
    df['length'] = df['问题名称'].apply(len)
    # 假设有一组文本长度数据
    text_lengths = df['length'].values.tolist()
    # 计算平均值和标准差
    mean_length = np.mean(text_lengths)
    std_dev = np.std(text_lengths)
    # 选择符合正态分布的数据
    x = 1  # 标准差倍数
    selected_data = [length for length in text_lengths if
                     mean_length - x * std_dev <= length <= mean_length + x * std_dev]
    # 从符合条件的数据中选择10条
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
    # shortest_data = text_lengths[:5]
    # longest_data = text_lengths[-5:]
    # select_list.extend(shortest_data)
    # select_list.extend(longest_data)
    # 从DataFrame中选择数据
    # 并且已经初始化了 select_list
    select_df_list = []
    for sel_num in select_list:
        selected_data = df.loc[df['length'] == sel_num].sample(n=1)
        select_df_list.append(selected_data)
    # 最后，将 select_list 中的所有结果合并成一个新的DataFrame
    combined_data = pd.concat(select_df_list)
    return combined_data


def select_rule(df):
    # 常见类别公积金、居住证、参保、企业,总计80条数据
    print(df.head())
    df_1 = df[df['业务标签'] == '公积金'].sample(1000)
    df_2 = df[df['业务标签'] == '居住证'].sample(1000)
    df_3 = df[df['业务标签'] == '参保'].sample(1000)
    df_4 = df[df['业务标签'] == '企业'].sample(1000)
    df_5 = df[df['业务标签'] == '其他'].sample(1000)
    common_df = pd.concat([df_1, df_2, df_3, df_4, df_5], ignore_index=True)
    # 正太分布取64条
    # 64+80
    zhengtai_df = len_text_min_max(common_df, n=140)
    # 每个类别取最大最小分别取两条
    df['length'] = df['问题名称'].apply(len)
    select_df_list = []
    for type in ['公积金', '居住证', '参保', '企业', '其他']:
        one_type = df[df['业务标签'] == type]
        text_lengths = one_type['length'].values.tolist()
        text_lengths.sort()
        shortest_data = text_lengths[:2]
        longest_data = text_lengths[-2:]
        for shortest in shortest_data:
            selected_data = one_type.loc[one_type['length'] == shortest].sample(n=1)
            select_df_list.append(selected_data)
        for shortest in longest_data:
            selected_data = one_type.loc[one_type['length'] == shortest].sample(n=1)
            select_df_list.append(selected_data)
    # 最后，将 select_list 中的所有结果合并成一个新的DataFrame
    combined_data = pd.concat(select_df_list)
    combined_data = pd.concat([combined_data, zhengtai_df])
    return combined_data


def qita_data():
    res_list = []
    sum_list = []
    with open(r'D:\work\资源\202202.txt', encoding='utf-8')as f:
        for line in f:
            contained_keywords = [keyword for keyword in keywords if keyword in line]
            if any(contained_keywords):
                res_list.append(line.strip())
                sum_list.append(','.join([str(i) for i in contained_keywords]))
            else:

                if len(line.strip()) > 0:
                    res_list.append(line.strip())
                    sum_list.append('其他')
    res_df = pd.DataFrame({'问题名称': res_list, '业务标签': sum_list})
    # 4份常见类别4份其他类别
    df_8 = select_rule(res_df)
    # 不常见类别直接加入
    res_df.loc[res_df['业务标签'].isin(['入伍', '军队', '参军']), '业务标签'] = '军队'
    select_df_list = []
    for type in ["公共服务", "海关", "教育", '军队']:
        one_type = res_df[res_df['业务标签'] == type].sample(n=10)
        select_df_list.append(one_type)
    combined_data = pd.concat(select_df_list)
    combined_data = pd.concat([combined_data, df_8])
    print(len(combined_data))
    print(combined_data['业务标签'].value_counts())
    # 对类别进行统计
    df_sorted = combined_data.sort_values(by='length')
    # 添加一个名为'分布'的新列，根据条件标记为'最长'、'最短'和'适中'
    df_sorted['分布'] = '适中'
    df_sorted.loc[df_sorted.index[:10], '分布'] = '最短'
    df_sorted.loc[df_sorted.index[-10:], '分布'] = '最长'
    df_sorted.to_excel('检验翻译繁体数据集.xlsx', index=False)
    # print(df_sorted.head())

    return res_df



from googletrans import Translator
translator = Translator(service_urls=['translate.google.com'])
def en_english(df):
    res_list = []
    for que in tqdm(df['问题名称'].values.tolist()):
        print(que)
        time.sleep(10)
        response = translator.translate(que, src='zh-cn', dest='en')
        print(response.text)
        res_list.append(response.text)
    df['英文'] = res_list
    df.to_excel('检验翻译繁体数据集_英繁.xlsx', index=False)
    return df


def en_fanti(df):
    import opencc
    cc = opencc.OpenCC('s2t')
    res_list = []
    for que in tqdm(df['问题名称'].values.tolist()):
        print(que)
        response = cc.convert(que)
        res_list.append(response)
    df['繁体'] = res_list
    return df


def fanti_jian(df):
    sel_list = []
    ori_list = []

    for que, fanti in tqdm(df[['问题名称', '繁体']].values.tolist()):
        flag = 0
        for i in que:
            if fanti.find(i) != -1:
                flag = 1
                break
        if flag == 1:
            sel_list.append(fanti)
            ori_list.append(que)
    df = pd.DataFrame({'问题': sel_list, '分析条件': ori_list, '类别': '简体+繁体'})
    print(len(df))
    print(df.head())
    return df


def jianti_yinwen(df):
    # 进行分词和词性标注
    que_n = []
    translate_n = []
    for que in tqdm(df['问题名称'].values.tolist()):
        words = pseg.cut(que)
        # 遍历分词结果
        translate = ''
        for word, flag in words:
            if flag == 'n':
                print('%s %s' % (word, flag))
                if len(word) > 0:
                    time.sleep(10)
                    response = fanyi.translate_text(word, to=fanyi.Lang.EN)
                    translate = str(que).replace(word, response)
        if len(translate) > 0:
            que_n.append(que)
            translate_n.append(translate)
    df = pd.DataFrame({'问题': que_n, '分析条件': translate_n, '类别': '简体+英文'})
    df.to_excel('检验翻译繁体数据集_简体翻译.xlsx', index=False)
    print(df.head())


def fanti_yinwen(df):
    # 进行分词和词性标注
    que_n = []
    translate_n = []
    fanti_list = []
    for que, fanti in tqdm(df[['问题名称', '繁体']].values.tolist()):
        words = pseg.cut(que)
        # 遍历分词结果
        translate = ''
        for word, flag in words:
            if flag == 'n':
                print('%s %s' % (word, flag))
                if len(word) > 0:
                    # 翻译的句子中找不到原始的词，说明这个是个简体，那就翻译简体
                    if fanti.find(word) != -1:
                        time.sleep(10)
                        try:
                            response = fanyi.translate_text(word, to=fanyi.Lang.EN)
                            translate = str(que).replace(word, response)
                        except:
                            print('出现无法翻译')
                            print(word)
        if len(translate) > 0:
            que_n.append(que)
            translate_n.append(translate)
            fanti_list.append(fanti)
    df = pd.DataFrame({'问题': que_n, '分析条件': translate_n, '繁体': fanti_list, '类别': '繁体+英文'})
    df.to_excel('检验翻译繁体数据集_繁体翻译.xlsx', index=False)
    print(df.head())


def configuration_data():
    # 构造200条全部转英文数据
    df = pd.read_excel('检验翻译繁体数据集.xlsx')
    # df = en_fanti(df)
    df = en_english(df)
    # 简体+繁体100条
    # fanti_jian(df)
    # 简体+英文
    # jianti_yinwen(df)
    # 繁体加英文
    # fanti_yinwen(df)
    # 简体_英文_繁体


configuration_data()
