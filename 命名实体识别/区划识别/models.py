import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch_crf import CRF


class Bert_CRF(nn.Module):  # BiLSTM加上并无多大用处,速度还慢了，可去掉LSTM层
    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.bert = BertModel.from_pretrained("/opt/modules/chinese-roberta-wwm-ext")
        # self.bert = BertModel.from_pretrained("D:\models\chinese-roberta-wwm-ext")
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(embedding_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size)

    def _get_features(self, sentence):
        with torch.no_grad():
            outputs = self.bert(sentence)
        enc = outputs.last_hidden_state
        enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test:  # Training，validation return loss
            loss = -self.crf.forward(emissions, tags, mask)
            return loss.mean()
        else:  # Testing，return decoding
            decode = self.crf.viterbi_decode(emissions, mask)
            return decode


class Bert_BiLSTM_CRF(nn.Module):  # BiLSTM加上并无多大用处,速度还慢了，可去掉LSTM层
    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        # self.bert = BertModel.from_pretrained("/opt/modules/chinese-roberta-wwm-ext")
        self.bert = BertModel.from_pretrained("D:\models\chinese-roberta-wwm-ext")
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        print('self.tagset_size')
        print(self.tagset_size)
        self.crf = CRF(self.tagset_size)

    def _get_features(self, sentence):
        with torch.no_grad():
            outputs = self.bert(sentence)
            embeds = outputs[0]
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats

    # def forward(self, sentence, tags, mask, is_test=True):
    #     emissions = self._get_features(sentence)
    #     if not is_test:  # Training，validation return loss
    #         loss = -self.crf.forward(emissions, tags, mask)
    #         return loss.mean()
    #     else:  # Testing，return decoding
    #         decode = self.crf.viterbi_decode(emissions, mask)
    #         return decode

    def forward(self, sentence, tags, mask):
        emissions = self._get_features(sentence)
        start_trans = self.crf.start_trans
        trans_matrix = self.crf.trans_matrix
        end_trans = self.crf.end_trans
        print(end_trans)
        return emissions, start_trans, trans_matrix, end_trans

    # def forward(self, sentence, tags, mask, is_test=True):
    #     emissions = self._get_features(sentence)
    #     decode = self.crf.viterbi_decode(emissions, mask)
    #     return str(decode)
