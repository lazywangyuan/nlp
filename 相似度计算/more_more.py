


# -*- coding: utf-8 -*-
import onnxruntime
import os
import numpy as np
from bert_input_process import format_simbert_input
from tokenization import Tokenizer
import joblib
with open('D:\work\simbert-master\simbert_old\cache-vecs-common-1705570013182.pkl', 'rb') as file_cache:
    new = joblib.load(file_cache)
# cpu
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
class Simbert():
    def __init__(self):
        self.max_seg_len = 128
        # CUDAExecutionProvider为gpu运行,CPUExecutionProvider为cpu运行
        self.sess = onnxruntime.InferenceSession(r"D:\models\shanghai\vec.onnx",
                                                 providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer(r"D:\models\shanghai\vec-vocab.txt", word_maxlen=self.max_seg_len,
                                   do_lower_case=True)

    def input_process(self, input_data):
        input_ids, segment_ids = format_simbert_input(input_data, max_seq_length=self.max_seg_len,
                                                      tokenizer=self.tokenizer)
        input_dict = {'Input-Token': input_ids, "Input-Segment": segment_ids}
        return input_dict

    def cal_semantic_sim(self,cal_vec,all_vec):
        """计算语义相似度"""
        vector = np.transpose(cal_vec, axes=[1, 0])
        cal_sims = np.dot(all_vec['vecs'], vector)
        trans_sims = cal_sims.T
        row_sims = trans_sims.reshape(1, -1)
        sims = row_sims[0]
        print(sims)
        return sims

    def transform_vector(self, text1,text2):
        text = [str(text1),str(text2)]
        input_feature1 = self.input_process(text)
        ret = self.sess.run(None, input_feature1)
        vecs = ret[0] / np.sum(ret[0] ** 2, axis=1, keepdims=True) ** 0.5
        return vecs

    def one_text(self):
        topn=5
        text1 = '外资企业或合资企业、民营企业可以申请专项么？'
        text2 = '国家财政如何对国家助学贷款利息进行补贴？'
        vecs1 = self.transform_vector(text1,text2)
        sims =self.cal_semantic_sim(vecs1,new)
        indexs = sims.argsort()[::-1]
        texts=[]
        for i in indexs:
            rel_key_index = i % len(new['keys'])
            print(rel_key_index)
            if sims[i] < 0.6 or len(texts) >= topn:
                break
            cache_text_key = new['keys'][rel_key_index]
            texts.append(cache_text_key)
            # 用于检查一个名为 texts 的列表中是否存在具有特定键值（cache_text_key）的字典。
            if list(filter(lambda x: x == cache_text_key, texts)):
                continue
        #     texts.append({"key": cache_text_key, "score": round(float(sims[i]), 4)})
        print(texts)



if __name__ == "__main__":
    model = Simbert()
    model.one_text()
