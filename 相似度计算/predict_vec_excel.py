# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import joblib
# 语义向量 onnx
import onnxruntime
import os
import numpy as np
from tqdm import tqdm
import joblib
from bert_input_process import format_simbert_input
from tokenization import Tokenizer

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# texts = []
# test_df = pd.read_excel(r'D:\work\model_nlp_llm\corpus\财政poc工单信息问答梳理20231031 v1.0.xlsx')
# for i in test_df['问题'].values.tolist():
#     texts.append(str(i).replace('\t', '').replace('\n', '').replace('\r', ''))

# with open('cache-vecs-common.pkl', 'rb') as file_cache:
#     new = joblib.load(file_cache)
print(datetime.datetime.now(), "head")
df = pd.read_csv('D:\work\simbert-master\common.csv')
texts = df['text'].values.tolist()
print(datetime.datetime.now(), "astype")
# texts = df["ST_ITEM_NAME"].str.cat(df["ST_SUBITEM_NAME"], sep="", na_rep="")


max_seg_len = 128
print(datetime.datetime.now(), "tokenization")
tokenizer = Tokenizer(r"/opt/modules/vec/vec-vocab.txt", word_maxlen=max_seg_len,
                      do_lower_case=True)


def input_process(input_data):
    input_ids, segment_ids = format_simbert_input(input_data, max_seq_length=max_seg_len,
                                                  tokenizer=tokenizer)
    input_dict = {'Input-Token': input_ids, "Input-Segment": segment_ids}
    return input_dict


print(datetime.datetime.now(), "model")
# sess = onnxruntime.InferenceSession(r"D:\work\model_nlp_vec_semantic\model\shanghai\vec.onnx",
#                                     providers=['CPUExecutionProvider'])
# gpu
# sess = onnxruntime.InferenceSession(r"D:\work\model_nlp_vec_semantic\model\shanghai\vec.onnx",
#                                     providers=['CUDAExecutionProvider'])

# sess = onnxruntime.InferenceSession(r"D:\work\model_nlp_vec_semantic\model\shanghai\vec.onnx",
#                                     providers=['CPUExecutionProvider'])
# gpu
sess = onnxruntime.InferenceSession(r"D:\work\simbert-master\simbert_old\simbert.onnx",
                                    providers=['CPUExecutionProvider'])

batch_size = 200
total = len(texts)


def predict(start):
    end = start + batch_size
    if end > total:
        end = total
    input_feature = input_process(texts[start:end])
    # onnx
    ret = sess.run(None, input_feature)
    vecs = ret[0]
    # tensorflow
    # ret=predict_fn(input_feature)
    # vecs = ret["Pooler-Dense"]
    vecs = vecs / np.sum(vecs ** 2, axis=1, keepdims=True) ** 0.5
    return vecs


print(datetime.datetime.now(), "start")
all_vec = [predict(start) for start in tqdm(range(0, total, batch_size))]
re_all_vec = np.vstack(all_vec)

with open("simbert_epoch2.npy", "wb") as file_cache:
    joblib.dump({"keys": texts, "vecs": re_all_vec}, file_cache)
