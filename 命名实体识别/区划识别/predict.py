# -*- coding: utf-8 -*-
from typing import Any, Union

import torch
from torch.utils import data
from models import Bert_BiLSTM_CRF
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import os

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
else:
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained("D:\models\chinese-roberta-wwm-ext")
ner_type = pd.read_excel("../data/bert/label.xlsx")  # 包含ner所有类别的txt文件
ners = ner_type["label"].tolist()
VOCAB = ners
VOCAB.extend(['<PAD>', '[CLS]', '[SEP]'])
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 128


class NerDataset_one(Dataset):
    ''' Generate our dataset '''

    def __init__(self, input, inference_df=None):
        self.sents = []
        self.tags_li = []
        for text in input:
            word = [i for i in text][:120]
            tag = ['O' for i in text][:120]
            self.sents.append(['[CLS]'] + word + ['[SEP]'])
            self.tags_li.append(['[CLS]'] + tag + ['[SEP]'])

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        token_ids = tokenizer.convert_tokens_to_ids(words)
        laebl_ids = [tag2idx[tag] for tag in tags]
        seqlen = len(laebl_ids)
        return token_ids, laebl_ids, seqlen

    def __len__(self):
        return len(self.sents)


def PadBatch(batch):
    maxlen = MAX_LEN
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    label_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask


def extract_entities_from_bio(tokens, bio_tags):
    entities = []
    current_entity = None
    current_type = None
    standard_dict = {'pro': 'province', 'market': "city", 'area': 'country', 'gov': 'government'}
    for token, tag in zip(tokens, bio_tags):
        if tag == '[SEP]':
            tag = 'O'
        if tag == '[CLS]':
            tag = 'O'
        if tag == 'O':
            if current_entity:
                entities.append((current_entity, standard_dict[current_type]))
                current_entity = None
                current_type = None
        elif tag.startswith('B-'):
            if current_entity:
                entities.append((current_entity, standard_dict[current_type]))
            current_entity = token
            current_type = tag.split('-')[1]
        elif tag.startswith('I-'):
            if current_entity and current_type == tag.split('-')[1]:
                current_entity += token
        else:
            raise ValueError("Invalid BIO tag: {}".format(tag))

    if current_entity:
        entities.append((current_entity, standard_dict[current_type]))

    return entities


def test(new_model, iterator, device):
    # loaded_paras = torch.load('model20.pt', map_location=torch.device('cpu'))
    loaded_paras = torch.load('model4_99.821.pt', map_location=torch.device('cpu'))
    new_model.load_state_dict(loaded_paras)  # 用本地已有模型来重新初始化网络权重参数
    new_model.eval()  # 注意不要忘记
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y, z = batch
            x = x.to(device)
            z = z.to(device)
            y_hat= new_model(x, y, z, is_test=True)
            print(y_hat)
            # Save prediction
            for j in y_hat:
                Y_hat.extend(j)
            # Save labels
            mask = (z == 1).cpu()
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig)

    Y = torch.cat(Y, dim=0).numpy()
    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]
    return y_true, y_pred


def input_test():
    while True:
        text = input('输入测试文本')
        test_dataset = NerDataset_one([text])
        test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=PadBatch)
        y_test, y_pred = test(model, test_iter, device)
        tokens = [i for i in text]
        bio_tags = y_pred[1:-1]
        print(extract_entities_from_bio(tokens, bio_tags))

def one_test():
    text = '我在成都双流区，我要去山东省济南市办理身份证'
    test_dataset = NerDataset_one([text])
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=PadBatch)
    y_test, y_pred = test(model, test_iter, device)
    tokens = [i for i in text]
    bio_tags = y_pred[1:-1]
    print(bio_tags)
    print(extract_entities_from_bio(tokens, bio_tags))

def df_test():
    df = pd.read_excel('../corpus/人工梳理易错数据.xlsx', sheet_name='Sheet1')
    # df = pd.read_excel('../coda/bert_crf_dev.xlsx')
    for text in tqdm(df['text'].values.tolist()):
        print(text)
        test_dataset = NerDataset_one([text])
        test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=PadBatch)
        y_test, y_pred = test(model, test_iter, device)
        tokens = [i for i in text]
        bio_tags = y_pred[1:-1]
        print(extract_entities_from_bio(tokens, bio_tags))


def df_test_df():
    df = pd.read_excel('../evaluate/完成区划评价指标.xlsx',sheet_name='Sheet2')
    # df = pd.read_excel('../coda/bert_crf_dev.xlsx')
    all_pre = []
    text_list = []
    for text in tqdm(df['text'].values.tolist()):
        print(text)
        text_list.append(text)
        test_dataset = NerDataset_one([text])
        test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=PadBatch)
        y_test, y_pred = test(model, test_iter, device)
        tokens = [i for i in text]
        bio_tags = y_pred[1:-1]
        print(tokens)
        print(bio_tags)
        all_pre.append(extract_entities_from_bio(tokens, bio_tags))
    df['pre'] = all_pre
    df['is_same'] = 0
    df.loc[df['label'] == df['pre'], 'is_same'] = 1
    df.to_excel('划评价指标结果_yuanshi.xlsx')


if __name__ == "__main__":
    model = Bert_BiLSTM_CRF(tag2idx)
    one_test()
