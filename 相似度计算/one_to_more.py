# -*- coding: utf-8 -*-
import numpy as np
import joblib
import datetime
import onnxruntime
import os
from tqdm import tqdm
from bert_input_process import format_simbert_input
from tokenization import Tokenizer
import json
# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
max_seg_len = 128
print(datetime.datetime.now(), "tokenization")
tokenizer = Tokenizer(r"D:\work\model_nlp_vec_semantic\model\shanghai\vec-vocab.txt", word_maxlen=max_seg_len,
                      do_lower_case=True)

def input_process(input_data):
    input_ids, segment_ids = format_simbert_input(input_data, max_seq_length=max_seg_len,
                                                  tokenizer=tokenizer)
    input_dict = {'Input-Token': input_ids, "Input-Segment": segment_ids}
    return input_dict


print(datetime.datetime.now(), "model")
sess = onnxruntime.InferenceSession(r"D:\work\model_nlp_vec_semantic\model\shanghai\vec.onnx",
                                    providers=['CPUExecutionProvider'])
# gpu
# sess = onnxruntime.InferenceSession(r"D:\work\model_nlp_vec_semantic\model\shanghai\vec.onnx",
#                                     providers=['CUDAExecutionProvider'])

with open(r'D:\work\simbert-master\simbert_old\train_pair_sim.npy', 'rb') as file_cache:
    all_vec = joblib.load(file_cache)


def get_most_sim(all_vecs, question_vec, topn=3, threshold=0.85):
    hits = []
    sims = np.dot(all_vecs, question_vec)
    indexs = sims.argsort()[::-1]
    score_list = []
    for index in indexs:
        # if sims[index] < threshold:
        #     break
        hits.append(index)
        score_list.append(sims[index])
        if len(hits) >= topn:
            break
    return hits, score_list

text1='汉娜，圣约翰先生终于说，这会儿就让她坐在那里吧，别问她问题。'
text1 = str(text1)
input_feature1 = input_process([text1])
ret = sess.run(None, input_feature1)
vecs = ret[0]
vecs1 = vecs / np.sum(vecs ** 2, axis=1, keepdims=True) ** 0.5
vecs_opposite = vecs1[0]
match_list = []
score_list = []
contrast_list = []
index_i, score = get_most_sim(all_vec['vecs'], vecs_opposite)
num = 0
for index, sco in zip(index_i, score):
    print(index)
    match_list.append(all_vec['keys'][index])
    score_list.append(sco)
print(match_list)
print(score_list)
