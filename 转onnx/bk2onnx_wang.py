#! -*- coding: utf-8 -*-
"""
环境：

232上使用tensorflow/tensorflow:1.15.4-gpu-py3镜像创建容器如下：

docker run --gpus='"device=0,1"' --restart=always --shm-size=16G -d --name=simbert -p8771:8991 -v /opt:/opt tensorflow/tensorflow:1.15.4-gpu-py3  /bin/sleep 10000000

/opt/nlp/new_simbert_train

第一次转saved-model，需要将环境安装如下：

bert4keras==0.10.6
keras==2.3.1
tensorflow-gpu==1.15.4
tensorflow-hub==0.12.0
h5py==2.10.0
然后运行bk2onnx.py文件
"""
import os

os.environ['TF_KERAS'] = '1'
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer

'''
pip install -i http://10.10.22.209:8090/repository/pypi/simple --trusted-host 10.10.22.209
生成saved-model的时候
bert4keras==0.10.6
keras==2.3.1
tensorflow-gpu==1.15.4
tensorflow-hub==0.12.0
h5py==2.10.0
'''
'''
生成后将saved-model转onnx，我是直接pip install tensorflow==2.5.0
建议创建一个新环境
bert4keras==0.10.6
keras==2.3.1
tensorflow==2.5.0
h5py==3.1.0
keras-nightly==2.5.0.dev2021032900
tensorflow-estimator==2.5.0
tf2onnx==1.9.1
onnx==1.9.0
onnxruntime==1.8.0
然后
# python -m tf2onnx.convert --saved-model encoder_model_tf --output encoder_simbert.onnx --opset 13
# python -m tf2onnx.convert --saved-model generate_model_tf --output generate_simbert.onnx --opset 13
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 基本信息
maxlen = 128

# 模型配置
model_file = '/opt/modules/chinese_simbert_L-12_H-768_A-12'
config_path = '{}/bert_config.json'.format(model_file)
checkpoint_path = '{}/bert_model.ckpt'.format(model_file)
# checkpoint_path = None
dict_path = '{}/vocab.txt'.format(model_file)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

roformer = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    application='unilm',
    with_pool='linear'
)
# 可以加载自己训练后的模型
# output_best_model = '/opt/nlp/new_simbert_train/output1/model_epoch20_loss1.6267908511191607.weights'
# if checkpoint_path is None:
#     roformer.load_weights(output_best_model)

# 向量生成模型
encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
encoder.save('model_path_epoch_ori/encoder_model_tf', save_format='tf')

# 解码器模型
seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])
outputs = [
    keras.layers.Lambda(lambda x: x[:, -1])(output)
    for output in seq2seq.outputs
]
generate_model = keras.models.Model(seq2seq.inputs, outputs)
generate_model.save('model_path_epoch_ori/generate_model_tf', save_format='tf')
print('完成')
# import keras2onnx
# onnx_model = keras2onnx.convert_keras(encoder)
# keras2onnx.save_model(onnx_model, 'bert-sim.onnx')
'''
经过验证这个时候需要tensorflow==2.5.0 
安装pip install tensorflow==2.5.0 
'''
'''
python -m tf2onnx.convert --saved-model model_path/encoder_model_tf --output model_path/simbert.onnx --opset 13
python -m tf2onnx.convert --saved-model model_path_epoch20/encoder_model_tf --output model_path_epoch20/simbert.onnx --opset 13

# 预测的时候还是用的tensorflow-gpu==1.15.4
如果是在一个环境里面操作，这里装两个tensorflow后再用bert4keras可能会出现问题，
uninstall tensorflow 后再install; h5py的版本也要注意。
pip uninstall keras-nightly
pip uninstall tensorflow
pip uninstall -y tensorflow-gpu
pip install keras==2.3.1
pip install tensorflow-gpu==1.15.4
pip install h5py==2.10.0
--opset 10 也是能保存成功，但是预测的时候出错了。
'''
#然后运行
#python -m tf2onnx.convert --saved-model model_path/encoder_model_tf --output model_path/simbert.onnx --opset 13