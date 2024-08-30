import gradio as gr
import json
import pandas as pd
from models import Bert_BiLSTM_CRF
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils import data

tokenizer = BertTokenizer.from_pretrained("/opt/modules/chinese-roberta-wwm-ext")
ner_type = pd.read_excel("label.xlsx")  # 包含ner所有类别的txt文件
ners = ner_type["label"].tolist()
VOCAB = ners
VOCAB.extend(['<PAD>', '[CLS]', '[SEP]'])
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 128
device = torch.device("cpu")


def clear():
    return ''


class Area():
    def __init__(self):
        self.model = Bert_BiLSTM_CRF(tag2idx)
        loaded_paras = torch.load('/opt/nlp/area_discern/simbert/BiLSTM_20240401_bz_128lr_0.0001/model10_99.846.pt',
                                  map_location=torch.device('cpu'))
        self.model.load_state_dict(loaded_paras)  # 用本地已有模型来重新初始化网络权重参数
        self.model.eval()  # 注意不要忘记
        self.provinces_dict = {}
        self.market_dict = {}
        self.area_dict = {}
        self.pma_dict = {}

    def cal_data(self):
        with open('province_contraction.json', 'r', encoding='utf-8') as load_f:
            province_contraction = json.load(load_f)
        for provinces in province_contraction:
            for province in province_contraction[provinces]:
                self.provinces_dict[province] = provinces

        with open('city_contraction.json', 'r', encoding='utf-8') as load_f:
            market_contraction = json.load(load_f)

        for markets in market_contraction:
            for market in market_contraction[markets]:
                self.market_dict[market] = markets

        df = pd.read_excel('省市区.xlsx')
        for provinces, market, area in df[['省', '市', '区']].values.tolist():
            self.pma_dict[area] = {'省': provinces, '市': market, '区': area}
            self.pma_dict[market] = {'省': provinces, '市': market}
            self.pma_dict[provinces] = {'省': provinces}

        for provinces, market, area, sam_provinces, sam_market, sam_area in df[
            ['省', '市', '区', '简称省', '简称市', '简称区']].values.tolist():
            self.provinces_dict[provinces] = provinces
            self.market_dict[market] = market
            self.area_dict[area] = area
            self.provinces_dict[sam_provinces] = provinces
            self.market_dict[sam_market] = market
            self.area_dict[sam_area] = area

    class NerDataset_one(Dataset):
        ''' Generate our dataset '''

        def __init__(self, input, inference_df=None):
            self.sents = []
            self.tags_li = []
            for text in input:
                word = [i for i in text][:120]
                tag = ['O' for i in text][:120]
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

    def PadBatch(self, batch):
        maxlen = MAX_LEN
        token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
        label_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in batch])
        mask = (token_tensors > 0)
        return token_tensors, label_tensors, mask

    def test(self, model, iterator, device):
        model.eval()  # 注意不要忘记
        Y, Y_hat = [], []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                x, y, z = batch
                x = x.to(device)
                z = z.to(device)
                y_hat = model(x, y, z, is_test=True)
                # Save prediction
                for j in y_hat:
                    Y_hat.extend(j)
                # Save labels
                mask = (z == 1).cpu()
                y_orig = torch.masked_select(y, mask)
                Y.append(y_orig)

        Y = torch.cat(Y, dim=0).numpy()
        y_true = [idx2tag[i] for i in Y]
        y_pred = [idx2tag[i] for i in Y_hat]
        return y_true, y_pred

    def extract_entities_from_bio(self, tokens, bio_tags):
        entities = []
        current_entity = None
        current_type = None
        standard_dict = {'pro': 'province', 'market': "city", 'area': 'country', 'gov': 'government'}
        for token, tag in zip(tokens, bio_tags):
            if tag == 'O':
                if current_entity:
                    entities.append((current_entity, standard_dict[current_type]))
                    current_entity = None
                    current_type = None
            elif tag.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, standard_dict[current_type]))
                current_entity = token
                current_type = tag.split('-')[1]
            elif tag.startswith('I-'):
                if current_entity and current_type == tag.split('-')[1]:
                    current_entity += token
            else:
                raise ValueError("Invalid BIO tag: {}".format(tag))

        if current_entity:
            entities.append((current_entity, standard_dict[current_type]))

        return entities

    def predict(self, text):
        test_dataset = self.NerDataset_one([text])
        test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=self.PadBatch)
        y_test, y_pred = self.test(self.model, test_iter, device)
        tokens = [i for i in text]
        bio_tags = y_pred[1:-1]
        res_data = self.extract_entities_from_bio(tokens, bio_tags)
        map_list = []
        result = []
        contains_gov = any(item[1].endswith('government') for item in res_data)
        if contains_gov:
            for item in res_data:
                if item[1] == 'province':
                    res_provinces = self.provinces_dict.get(item[0], '无')
                    if res_provinces != '无':
                        map_list.append(self.pma_dict.get(res_provinces, '无'))
                if item[1] == 'city':
                    res_provinces = self.market_dict.get(item[0], '无')
                    if res_provinces != '无':
                        map_list.append(self.pma_dict.get(res_provinces, '无'))
                if item[1] == 'country':
                    res_provinces = self.area_dict.get(item[0], '无')
                    if res_provinces != '无':
                        map_list.append(self.pma_dict.get(res_provinces, '无'))
            for i in range(len(map_list)):
                include = False
                # 检查当前字典是否被其他字典包含
                for j in range(len(map_list)):
                    if i != j and all(item in map_list[j].items() for item in map_list[i].items()):
                        include = True
                        break
                # 如果当前字典不被其他字典包含，则添加到最终结果列表中
                print(map_list)
                if not include:
                    result.append({'省': map_list[i].get('省', '无'), '市': map_list[i].get('市', '无'),
                                   '区': map_list[i].get('区', '无')})
        else:
            result.append({'省': '无', '市': '无', '区': '无'})
        if len(result) == 0:
            result.append({'省': '无', '市': '无', '区': '无'})
        print(result)
        df = pd.DataFrame(result)
        print(df)
        return res_data, df


model = Area()
model.cal_data()

with gr.Blocks() as demo:
    with gr.Row():
        title = gr.Textbox(label="文本")
    with gr.Row():
        output_ori = gr.Textbox(label="原始分析结果")
    with gr.Row():
        output = gr.DataFrame(headers=['规则处理后模型识别出的省市区'])
    with gr.Row():
        greet_btn = gr.Button("提交")
        greet_btn.click(fn=model.predict, inputs=[title], outputs=[output_ori, output])
        greet_btn = gr.Button("清除")
        greet_btn.click(clear, outputs=[title], show_progress=True)

demo.title = '区划识别'
# demo.launch()
# demo.launch(server_name='0.0.0.0',server_port=7863, app_name="/policy", share=False, inbrowser=True, show_api=False)
# demo.launch(server_name='127.0.0.1', server_port=7863, share=False, inbrowser=True)
demo.launch(server_name='10.10.22.232', server_port=7895, share=False, inbrowser=True)
