import joblib
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


class PocModel():
    def __init__(self):
        with open('simbert.npy', 'rb') as file_cache:
            self.all_vec = joblib.load(file_cache)
        self.max_seg_len = 128
        print(datetime.datetime.now(), "tokenization")
        self.tokenizer = Tokenizer(r"D:\work\model_nlp_vec_semantic\model\shanghai\vec-vocab.txt",
                                   word_maxlen=self.max_seg_len,
                                   do_lower_case=True)
        self.sess = onnxruntime.InferenceSession(r"D:\work\model_nlp_vec_semantic\model\shanghai\vec.onnx",
                                                 providers=['CPUExecutionProvider'])
        self.batch_size = 200

    def input_process(self, input_data):
        input_ids, segment_ids = format_simbert_input(input_data, max_seq_length=self.max_seg_len,
                                                      tokenizer=self.tokenizer)
        input_dict = {'Input-Token': input_ids, "Input-Segment": segment_ids}
        return input_dict

    def predict(self, start, total, texts):
        end = start + self.batch_size
        if end > total:
            end = total
        input_feature = self.input_process(texts[start:end])
        # onnx
        ret = self.sess.run(None, input_feature)
        vecs = ret[0]
        # tensorflow
        # ret=predict_fn(input_feature)
        # vecs = ret["Pooler-Dense"]
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        # vecs = vecs / np.sum(vecs ** 2, axis=1, keepdims=True) ** 0.5
        return vecs

    def simbert_text(self, texts):
        total = len(texts)
        print(datetime.datetime.now(), "start")
        all_vec = [self.predict(start, total, texts) for start in tqdm(range(0, total, self.batch_size))]
        re_all_vec = np.vstack(all_vec)
        return re_all_vec

    def get_most_sim(self, all_vecs, question_vec, topn=3, threshold=0.85):
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

    def search_diff(self, re_all_vec, text):
        que_list = []
        match_list = []
        score_list = []
        index_list = []
        for num, vecs_opposite in enumerate(re_all_vec):
            # print(get_most_sim(all_['vecs'], vecs_opposite))
            index_i, score = self.get_most_sim(self.all_vec['vecs'], vecs_opposite)
            num = 0
            for index, sco in zip(index_i, score):
                que_list.append(text)
                match_list.append(self.all_vec['keys'][index])
                score_list.append(sco)
                index_list.append(num)
                num = num + 1
        return que_list, match_list, score_list, index_list

    def read_data(self):
        df = pd.read_excel('../对比_简体转化繁体.xlsx',sheet_name='英文')
        que_list = []
        match_list = []
        score_list = []
        index_list = []
        for que in tqdm(df['问题名称'].values.tolist()):
            texts = que
            re_all_vec = PocModel.simbert_text([texts])
            que, match, score, index = PocModel.search_diff(re_all_vec, texts)
            que_list.extend(que)
            match_list.extend(match)
            score_list.extend(score)
            index_list.extend(index)
        pd.DataFrame({'问题名称': que_list, '相似问题': match_list, '相似度': score_list, '顺序': index_list}) \
            .to_excel('结果_simbert_英文相似.xlsx', index=False)

    def read_top1_data(self):
        df = pd.read_excel('包含正文相似_fanyi1.xlsx',sheet_name='Sheet1')
        que_list = []
        match_list = []
        score_list = []
        index_list = []
        for que, fanti in tqdm(df[['问题名称', '英文']].values.tolist()):
            texts = fanti
            re_all_vec = PocModel.simbert_text([texts])
            que, match, score, index = PocModel.search_diff(re_all_vec, texts)
            que_list.append(que[0])
            match_list.append(match[0])
            score_list.append(score[0])
        print(len(que_list))
        pd.DataFrame({'问题名称': que_list, '相似问题': match_list, '相似度': score_list}) \
            .to_excel('相似后的句子再转化为英文做相似.xlsx', index=False)


# def top2_fanyi():


PocModel = PocModel()
PocModel.read_data()
# texts = ['继续教育']
# re_all_vec = PocModel.simbert_text(texts)
# PocModel.search_diff(re_all_vec,texts)
