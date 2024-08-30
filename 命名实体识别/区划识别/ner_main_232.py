# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import argparse
import numpy as np
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from models import Bert_BiLSTM_CRF
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import os

# 检查是否有可用的GPU
if torch.cuda.is_available():
    print('GPU')
    device = torch.device("cuda")
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
else:
    print('cpu')
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained("/opt/modules/chinese-roberta-wwm-ext")
ner_type = pd.read_excel("data/label.xlsx")  # 包含ner所有类别的txt文件
ners = ner_type["label"].tolist()
VOCAB = ners
VOCAB.extend(['<PAD>', '[CLS]', '[SEP]'])
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 128


class NerDataset(Dataset):
    ''' Generate our dataset '''

    def __init__(self, f_path, inference_df=None):
        self.sents = []
        self.tags_li = []
        if inference_df is not None:
            data = inference_df
        else:
            data = pd.read_csv(f_path,nrows=10000)
        for word_str, tag_str in data[['text', 'label']].values.tolist():
            word = [i for i in word_str][:120]
            # 转换为列表
            tag = eval(tag_str)[:120]
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


def train(e, model, iterator, optimizer, scheduler, criterion, device):
    model.train()
    losses = 0.0
    step = 0
    for i, batch in enumerate(iterator):
        step += 1
        x, y, z = batch
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        loss, _ = model(x, y, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses += loss.item()
    print("Epoch: {}, Loss:{:.4f}".format(e, losses / step))


def validate(e, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            step += 1
            x, y, z = batch
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            loss, y_hat = model(x, y, z)
            losses += loss.item()
            # Save prediction
            for j in y_hat:
                Y_hat.extend(j)
            # Save labels
            mask = (z == 1)
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig.cpu())

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean() * 100

    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]

    print(metrics.classification_report(y_true, y_pred, labels=labels, digits=3))
    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(e, losses / step, acc))
    return model, losses / step, acc


if __name__ == "__main__":
    ner_type = pd.read_excel("data/label.xlsx")
    ners = ner_type["label"].tolist()
    labels = ners
    print("all type len is {}".format(len(labels)))
    best_model = None
    _best_val_loss = np.inf
    _best_val_acc = -np.inf

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--trainset", type=str, default="data/bert_crf_train.csv")
    parser.add_argument("--validset", type=str, default="data/bert_crf_dev.csv")
    parser.add_argument("--testset", type=str, default="data/bert_crf_dev.csv")

    ner = parser.parse_args()
    model = Bert_BiLSTM_CRF(tag2idx).to(device)
    print('Initial model Done.')
    train_dataset = NerDataset(ner.trainset)
    print("train data len is {}".format(len(train_dataset)))
    eval_dataset = NerDataset(ner.validset)
    print("validset data len is {}".format(len(eval_dataset)))
    test_dataset = NerDataset(ner.testset)
    print("test_dataset len is {}".format(len(test_dataset)))
    print('Load Data Done.')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=ner.batch_size,
                                 shuffle=True,
                                 collate_fn=PadBatch)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=ner.batch_size,
                                shuffle=False,
                                collate_fn=PadBatch)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=ner.batch_size,
                                shuffle=False,
                                collate_fn=PadBatch)
    optimizer = AdamW(model.parameters(), lr=ner.lr, eps=1e-6)
    len_dataset = len(train_dataset)
    epoch = ner.n_epochs
    batch_size = ner.batch_size
    total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (
                                                                                                    len_dataset // batch_size + 1) * epoch
    warm_up_ratio = 0.1  # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in tqdm(range(1, ner.n_epochs + 1)):
        train(epoch, model, train_iter, optimizer, scheduler, criterion, device)
        candidate_model, loss, acc = validate(epoch, model, eval_iter, device)
        torch.save(model.state_dict(),
                   'BiLSTM_202404017_bz_128lr_0.0005/' + 'model' + str(epoch) + '_' + str(round(acc, 3)) + '.pt')
        if loss < _best_val_loss and acc > _best_val_acc:
            best_model = candidate_model
            _best_val_loss = loss
            _best_val_acc = acc
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # y_test, y_pred = test(model, test_iter, device)
    # print(metrics.classification_report(y_test, y_pred, labels=labels, digits=3))
