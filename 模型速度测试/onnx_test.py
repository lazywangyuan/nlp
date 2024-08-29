# 测试在cpu、gpu上的表现
"""
    1分钟能跑多个个样本呢
    按照正太分布取最长的10条，最短的10条，中间的10条
    同一个长度取10条

    1个样本需要多少时间

    模型大小
"""
import time
import os
from bert_input_process import format_simbert_input
from tokenization import Tokenizer
import onnxruntime

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
max_seg_len = 128
# tokenizer = Tokenizer(r"D:\work\model_nlp_vec_semantic\model\shanghai\vec-vocab.txt", word_maxlen=max_seg_len,
#                       do_lower_case=True)
# sess = onnxruntime.InferenceSession(r"D:\work\simbert-master\simbert_old\simbert.onnx",
#                                     providers=['CPUExecutionProvider'])
# 232
tokenizer = Tokenizer("/opt/modules/vec/vec-vocab.txt", word_maxlen=max_seg_len,
                      do_lower_case=True)
sess = onnxruntime.InferenceSession("/opt/modules/vec/vec.onnx",
                                    providers=['CPUExecutionProvider'])


def input_process(input_data):
    input_ids, segment_ids = format_simbert_input(input_data, max_seq_length=max_seg_len,
                                                  tokenizer=tokenizer)
    input_dict = {'Input-Token': input_ids, "Input-Segment": segment_ids}
    return input_dict


# 模拟处理一个样本的函数
def process_sample(sample):
    input_feature1 = input_process([sample])
    ret = sess.run(None, input_feature1)
    return ret


# 模拟获取样本的函数
def get_sample():
    # 这里可以填入你的样本获取代码
    return '第十四届陆家嘴论坛6月8日开幕 李云泽龚正担任共同轮值主席邀请70余名中外演讲嘉宾'


# 测试一分钟内处理多少个样本
def test_throughput():
    start_time = time.time()
    samples_processed = 0
    while time.time() - start_time < 60:
        print(samples_processed)
        sample = get_sample()
        process_sample(sample)
        samples_processed += 1

    print("在一分钟内处理了 %d 个样本。" % samples_processed)


if __name__ == "__main__":
    test_throughput()
