import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm
from sklearn import metrics
from torch.nn import functional as F


def load_dataset(path):
    data = []
    label_dict = {'others': 0, 'happy': 1, 'sad': 2, 'angry': 3}
    with open(path, 'r', encoding='UTF-8') as fp:
        next(fp)  # 跳过首行
        for line in tqdm(fp.readlines()):
            line = line.strip()
            sentence1, sentence2, sentence3, label = line.split('\t')
            label = label_dict.get(label, -1)
            data.append([sentence1, sentence2, sentence3, label])
    return data


class myDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sent1, sent2, sent3, label = self.data[item]
        encoding = self.tokenizer.batch_encode_plus(
            [sent1, sent2, sent3],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'conversation_text': sent1 + sent2,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertSentenceEmbeddings(nn.Module):
    def __init__(self, freeze_embedding=True):
        super(BertSentenceEmbeddings, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.freeze_embedding = freeze_embedding

    def forward(self, input_ids, attention_mask):
        if self.freeze_embedding:
            with torch.no_grad():
                output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output[0]
        return cls_output

# 4. 整合BERT和CNN
class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()
        self.bert_embeddings = BertSentenceEmbeddings()
        self.lstm = nn.LSTM(input_size=768, hidden_size=300, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(300 * 2, 200),
                                nn.BatchNorm1d(200),
                                nn.Linear(200, num_classes))

    def attention_net(self, lstm_output, final_state):
        # hidden:[16, 600]->[16, 600, 1]
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)

        # [16, 768, 600] * [16, 600, 1] = [16, 768, 1] ->[16, 768]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)

        # attn_weights : [batch_size,n_step]=[16, 768]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # lstm_output:[16, 768, 600] -> [16, 600, 768]
        # soft_attn_weights: [16, 768] -> [16, 768, 1]
        # context: [16, 600]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights

    def forward(self, input_ids, attention_mask):
        embeddings = []
        for i in range(3):  # 对每个句子处理
            sentence_embedding = self.bert_embeddings(input_ids[:, i, :], attention_mask[:, i, :])
            embeddings.append(sentence_embedding)

        embeddings = torch.cat(embeddings, dim=1)  # 拼接三个[16，256, 768] ->[16，768, 768]
        out, (hs, _) = self.lstm(embeddings)  # out: [16, 768, 600], hs: [4, 16, 300]
        att_out, attention = self.attention_net(out, hs)
        out = self.fc(att_out)
        return out


