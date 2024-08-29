from paddlenlp.datasets import load_dataset


def load_data(data_path):
    with open(data_path, 'r') as f:
        for item in f.readlines():
            query, title, label = item.split('\t')
            yield {'text_a': query, 'text_b': title, 'label': int(label)}


# 一键加载 LCQMC 的训练集、验证集
train_dataset = load_dataset(load_data, data_path='train_sim.tsv', lazy=False)
dev_dataset = load_dataset(load_data, data_path='train_sim.tsv', lazy=False)
test_dataset = load_dataset(load_data, data_path='train_sim.tsv', lazy=False)

print('训练集样本数：', len(train_dataset))
print('训练集样本数：', len(dev_dataset))
print('训练集样本数：', len(test_dataset))
print('样本示例：', train_dataset[0])

import numpy as np


def convert_example_to_feature(example,
                               tokenizer,
                               max_seq_length=512,
                               is_test=False,
                               is_pair=False):
    if is_pair:
        # 句子a
        text = example["text_a"]
        # 句子b
        text_pair = example["text_b"]
    else:
        text = example["text"]
        text_pair = None
    # 将句子a和句子b输入到tokenizer中转换成ID的形式
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair,
                               max_seq_len=max_seq_length)

    if is_test:
        return encoded_inputs
    # 将label转换成numpy的形式
    encoded_inputs["label"] = np.array([example["label"]], dtype="int64")
    return encoded_inputs


from paddlenlp.transformers import AutoTokenizer
from functools import partial

tokenizer = AutoTokenizer.from_pretrained("rocketqa-zh-base-query-encoder")
max_seq_length = 128
# 给convert_example_to_feature函数设置默认值
trans_func = partial(convert_example_to_feature,
                     tokenizer=tokenizer,
                     max_seq_length=max_seq_length,
                     is_pair=True)
# 把训练集，验证集，测试集映射成ID的形式
train_dataset = train_dataset.map(trans_func)
dev_dataset = dev_dataset.map(trans_func)
test_dataset = test_dataset.map(trans_func)

from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import DataCollatorWithPadding

# 文本填充
data_collator = DataCollatorWithPadding(tokenizer)
batch_size = 64
# 训练集，训练集设置shuffle为True，进行打乱数据操作
train_sampler = BatchSampler(
    train_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_sampler=train_sampler,
                          collate_fn=data_collator)
# 验证集，验证集和测试集不需要对数据进行打乱，则设置shuffle为False
dev_sampler = BatchSampler(
    dev_dataset, batch_size=batch_size, shuffle=False)
dev_loader = DataLoader(dataset=dev_dataset,
                        batch_sampler=dev_sampler,
                        collate_fn=data_collator)
# 测试集
test_sampler = BatchSampler(
    test_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset,
                         batch_sampler=test_sampler,
                         collate_fn=data_collator)

import paddle.nn as nn
import paddle.nn.functional as F


class CrossEncoder(nn.Layer):

    def __init__(self, pretrained_model, dropout=None, num_classes=2):
        """
        CrossEncoder的框架的实现
        输入：
            - pretrained_model：预训练语言模型
            - dropout：dropout的参数
            - num_classes：类别数目，对于语义匹配任务而言，一般是二分类。
        """
        super().__init__()
        self.ernie = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes,
                                    weight_attr=nn.initializer.TruncatedNormal(
                                        mean=0.0, std=0.02),
                                    bias_attr=nn.initializer.Constant(value=0))

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                encoder_type='first-last-avg'):
        '''
            :param input_ids:
            :param attention_mask:
            :param encoder_type: encoder_type:  "first-last-avg""
            :return:
        '''
        sequence_output, pooled_output, hidden_output = self.ernie(input_ids,
                                                                   token_type_ids=token_type_ids,
                                                                   position_ids=position_ids,
                                                                   attention_mask=attention_mask,
                                                                   output_hidden_states=True)
        if encoder_type == 'first-last-avg':  # average of the first and the last layers.
            first = hidden_output[1]  # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            last = hidden_output[-1]
            seq_length = first.shape[1]
            first_avg = F.avg_pool1d(first.transpose([0, 2, 1]), kernel_size=seq_length).squeeze(-1)
            last_avg = F.avg_pool1d(last.transpose([0, 2, 1]), kernel_size=seq_length).squeeze(-1)
            final_encoding = F.avg_pool1d(
                paddle.concat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], axis=1).transpose([0, 2, 1]),
                kernel_size=2).squeeze(-1)
        logits = self.classifier(final_encoding)
        return logits


from paddlenlp.transformers import AutoModel
import paddle

# epoch 参数，可以根据情况自行调节
num_epochs = 1
learning_rate = 5e-5
eval_steps = 50
log_steps = 10
weight_decay = 0.0
save_dir = "./checkpoints"

model_name_or_path = "rocketqa-zh-base-query-encoder"
# 类别数目
num_classes = 2
# 预训练语言模型
pretrained_model = AutoModel.from_pretrained(model_name_or_path)
# CrossEncoder模型
model = CrossEncoder(pretrained_model, num_classes=num_classes)

# 所有的norm和bias的参数不需要weight decay
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 设置优化器
optimizer = paddle.optimizer.AdamW(
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params,
    learning_rate=learning_rate)
# 交叉熵损失函数
loss_fn = paddle.nn.CrossEntropyLoss()
# Accuracy 评估方式
metric = paddle.metric.Accuracy()


def evaluate(model, data_loader, metric):
    # 将模型设置为评估模式
    model.eval()
    # 重置metric
    metric.reset()

    # 遍历验证集每个批次
    for batch_id, data in enumerate(dev_loader):
        input_ids, token_type_ids, labels = data["input_ids"], data["token_type_ids"], data["labels"]
        # 计算模型输出
        logits = model(input_ids, token_type_ids)
        # 累积评价
        correct = metric.compute(logits, labels)
        metric.update(correct)

    dev_score = metric.accumulate()

    return dev_score


import os


def train(model):
    # 开启模型训练模式
    model.train()
    global_step = 0
    best_score = 0.
    # 记录训练过程中的loss 和 在验证集上模型评估的分数
    train_loss_record = []
    train_score_record = []
    num_training_steps = len(train_loader) * num_epochs
    # 进行num_epochs轮训练
    for epoch in range(num_epochs):
        for step, data in enumerate(train_loader):
            input_ids, token_type_ids, labels = data["input_ids"], data["token_type_ids"], data["labels"]
            # 获取模型预测s
            logits = model(input_ids, token_type_ids)
            loss = loss_fn(logits, labels)  # 默认求mean
            train_loss_record.append((global_step, loss.item()))

            # 梯度反向传播
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if global_step % log_steps == 0:
                print(
                    f"[Train] epoch: {epoch}/{num_epochs}, step: {global_step}/{num_training_steps}, loss: {loss.item():.5f}")

            if global_step != 0 and (global_step % eval_steps == 0 or global_step == (num_training_steps - 1)):
                dev_score = evaluate(model, dev_loader, metric)
                train_score_record.append((global_step, dev_score))
                print(f"[Evaluate]  dev score: {dev_score:.5f}")
                model.train()

                # 如果当前指标为最优指标，保存该模型
                if dev_score > best_score:
                    save_path = os.path.join(save_dir, "best.pdparams")
                    paddle.save(model.state_dict(), save_path)
                    print(
                        f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                    best_score = dev_score

            global_step += 1

    save_path = os.path.join(save_dir, "final.pdparams")
    paddle.save(model.state_dict(), save_path)
    print("[Train] Training done!")

    return train_loss_record, train_score_record


# train_loss_record, train_score_record = train(model)

# 加载训练好的模型进行预测，重新实例化一个模型，然后将训练好的模型参数加载到新模型里面
saved_state = paddle.load("./checkpoints/best.pdparams")
model = CrossEncoder(pretrained_model)
model.load_dict(saved_state)
# 评估模型
evaluate(model, test_loader, metric)


def infer(model, tokenizer, query, title):
    label_map = {0: '不相似', 1: '相似'}
    # 编码映射为id
    encoded_inputs = tokenizer(query, text_pair=title)
    input_ids = paddle.to_tensor([encoded_inputs["input_ids"]])
    token_type_ids = paddle.to_tensor([encoded_inputs["token_type_ids"]])
    # 调用模型
    model.eval()
    logits = model(input_ids, token_type_ids)
    probs = F.softmax(logits, axis=1)
    idx = paddle.argmax(probs, axis=1).numpy()
    idx = idx.tolist()
    pred_label = label_map[idx[0]]
    print('Label: {}'.format(pred_label))


# 输入一条样本
query = '世界上什么东西最小'
title = '世界上什么东西最小？'
infer(model, tokenizer, query, title)
