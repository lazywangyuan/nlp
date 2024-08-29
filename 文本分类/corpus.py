import pandas as pd


def split_data(test_df, group_name):
    grouped = test_df.groupby(group_name)
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for _, group in grouped:
        num_samples = len(group)
        if num_samples > 1:
            num_train_samples = int(num_samples * 0.7)
            num_test_samples = num_samples - num_train_samples
            # 随机打乱样本顺序
            shuffled_group = group.sample(frac=1, random_state=20230628).reset_index(drop=True)
            add_group = shuffled_group[(shuffled_group['is_add'] == 1) | (shuffled_group['is_add'] == '1')]
            train_samples = shuffled_group[:num_train_samples]
            test_samples = shuffled_group[num_train_samples:num_train_samples + num_test_samples]
            print(train_samples.head())
            train_set = pd.concat([train_set, train_samples], ignore_index=True)
            # train_set = train_set.append(train_samples)
            train_set = pd.concat([train_set, add_group], ignore_index=True)
            # train_set = train_set.append(add_group)
            train_set = train_set.drop_duplicates(['问题名称', group_name])
            test_set = pd.concat([test_set, test_samples], ignore_index=True)
            # test_set = test_set.append(test_samples)
        else:
            shuffled_group = group.sample(frac=1, random_state=20230628).reset_index(drop=True)
            train_set = pd.concat([train_set, shuffled_group], ignore_index=True)
            # train_set = train_set.append(shuffled_group)
            test_set = pd.concat([test_set, shuffled_group], ignore_index=True)
            # test_set = test_set.append(shuffled_group)
    train_set = train_set.drop_duplicates(subset=['问题名称'])
    test_set = test_set.drop_duplicates(subset=['问题名称'])
    return train_set, test_set


test_df = pd.read_excel('演示数据.xlsx')
train_set, test_set = split_data(test_df, '业务标签')
file = open("data/train/train.txt", "w", encoding='utf-8')
for que, label in train_set[['问题名称', '业务标签']].values:
    print(que)
    file.write(str(que).replace('\t', '').replace('\n', '').replace('\r', '') + '\t' + label + '\n')

file = open("data/test/dev.txt", "w", encoding='utf-8')
for que, label in test_set[['问题名称', '业务标签']].values:
    print(que)
    file.write(str(que).replace('\t', '').replace('\n', '').replace('\r', '') + '\t' + label + '\n')

file = open("data/train/data.txt", "w", encoding='utf-8')
for que, label in test_df[['问题名称', '业务标签']].values:
    print(que)
    file.write(str(que).replace('\t', '').replace('\n', '').replace('\r', '') + '\n')

label_list = []
for i in train_set['业务标签'].values.tolist():
    label_list.append(i)
label_list = sorted(set(label_list))
# label_list = sorted(set([str(i).split(',') for i in short_df['最终目录'].values.tolist]))
file = open("data/train/label.txt", "w", encoding='utf-8')
# # 写入文本内容到文件中
for label in label_list:
    file.write(label + '\n')
