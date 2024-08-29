import gradio as gr
from paddlenlp.transformers import ErnieTokenizer
import onnxruntime
import numpy as np

class KnTopic():
    def __init__(self):
        f = open('model\label.txt', encoding='utf-8')
        self.label_list = [i for i in f.read().split('\n')]
        self.cal_tokenizer = ErnieTokenizer.from_pretrained('model\paddle')
        self.cal_sess = onnxruntime.InferenceSession('model\paddle\cls_kn_topic.onnx')
        print('模型加载完成')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, text):
        encoded_inputs = self.cal_tokenizer(text=text, max_seq_len=512, pad_to_max_seq_len=False,
                                            truncation_strategy="longest_first")
        pred = self.cal_sess.run(None, {
            self.cal_sess.get_inputs()[0].name: np.array([encoded_inputs['input_ids']]).astype(np.int64),
            self.cal_sess.get_inputs()[1].name: np.array([encoded_inputs['token_type_ids']]).astype(np.int64)})[0]
        probs = self.sigmoid(pred[0])
        labels = []
        score = []
        for i, p in enumerate(probs):
            if p > 0.9:
                labels.append(self.label_list[i])
                score.append(p)
        if len(score) > 0:
            max_score = str(max(score))
            max_label = labels[score.index(max(score))]
        else:
            max_score = '1'
            max_label = '其他'
        return max_score, max_label


model = KnTopic()


def greet(text):
    print(text)
    if text.strip() == "":
        return "内容不能为空，请重新输入。"
    max_score, max_label = model.predict(text)
    print(max_score, max_label)
    res = '情绪分类: ' + max_label + '\n' + '置信度: ' + max_score
    print(res)
    return str(res)


def clear(*args):
    return ""


with gr.Blocks() as demo:
    title = gr.Textbox(label="情绪")
    output = gr.Textbox(label="分析结果")
    greet_btn = gr.Button("提交")
    greet_btn.click(fn=greet, inputs=[title], outputs=output)
    greet_btn = gr.Button("清除")
    for clear_text in [title, output]:
        greet_btn.click(fn=clear, inputs=clear_text, outputs=clear_text)

demo.title = '情绪分类'
# demo.launch()
# demo.launch(server_name='0.0.0.0',server_port=7863, app_name="/policy", share=False, inbrowser=True, show_api=False)
# demo.launch(server_name='127.0.0.1', server_port=7863, share=False, inbrowser=True)
demo.launch(server_name='10.10.22.232', server_port=7868, share=False, inbrowser=True)