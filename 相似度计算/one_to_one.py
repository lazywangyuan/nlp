# -*- coding: utf-8 -*-
import onnxruntime
import os
import numpy as np
from bert_input_process import format_simbert_input
from tokenization import Tokenizer

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

    def transform_vector(self, text):
        text = [str(text)]
        input_feature1 = self.input_process(text)
        ret = self.sess.run(None, input_feature1)
        vecs = ret[0]
        vecs = vecs / np.sum(vecs ** 2, axis=1, keepdims=True) ** 0.5
        return vecs

    def one_text(self):
        text1 = '大猩猩过冬储粮'
        text2 = '大猩猩过冬储'
        vecs1 = self.transform_vector(text1)
        vecs1 = vecs1[0]
        vecs2 = self.transform_vector(text2)
        vecs2 = vecs2[0]
        sims = np.dot(vecs1, vecs2)
        print(sims)


if __name__ == "__main__":
    model = Simbert()
    model.one_text()
