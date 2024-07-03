# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 层次化的LSTM,对三句话分别做LSTM，再将得到的向量拼接，再用LSTM进行特征提取，最后放入全连接网络分类。

class lstm2(nn.Module):
    def __init__(self):
        super(lstm2, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(
            np.load('./embedding_SougouNews.npz')["embeddings"].astype('float32')), freeze=False)

        self.lstm1 = nn.LSTM(300, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(300, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        # self.lstm3 = nn.LSTM(300, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.lstm4 = nn.LSTM(128*6, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128 *2, 32)
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x1, x2, x3 = x[0], x[1], x[2]
        # x0, x1, x2分别为第1,2,3句话。
        out1 = self.embedding(x1)  # [batch_size, seq_len, embeding]
        # print(out1.shape)

        out1, (hs1,_) = self.lstm1(out1)

        out2 = self.embedding(x2)
        out2, (hs2,_) = self.lstm2(out2)
        out3 = self.embedding(x3)
        out3, (hs3,_) = self.lstm1(out3)
        out_all = torch.cat([out1, out2, out3], dim=-1)
        out,(hs,_) = self.lstm4(out_all)
        hs = torch.cat([hs[-1,:,:],hs[-2,:,:]],dim=-1)   # 得到最后时刻的隐藏状态
        # print(hs.shape)
        out = self.fc(hs)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


#仅使用一个LSTM，对最后一句话进行分析
class lstm1(nn.Module):
    def __init__(self):
        super(lstm1, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(
            np.load('./embedding_SougouNews.npz')["embeddings"].astype('float32')), freeze=False)

        self.lstm = nn.LSTM(300, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(2 * 128, 32)
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = x[2]
        # x0, x1, x2分别为第1,2,3句话。

        out = self.embedding(x)  # [batch_size, seq_len, embeding]

        out, (hs1,_) = self.lstm(out)

        out = self.fc(out[:, -1, :])
        # out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

#层次化的LSTM + attention
class LSTM_att(nn.Module):
    def __init__(self):
        super(LSTM_att, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(
            np.load('./embedding_SougouNews.npz')["embeddings"].astype('float32')), freeze=False)

        self.lstm1 = nn.LSTM(300, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(300, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        # self.lstm3 = nn.LSTM(300, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.lstm4 = nn.LSTM(768, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(128 * 2, 32)
        self.fc2 = nn.Linear(32, 4)

        self.fc3 = nn.Linear(128*2*3, 128)
        self.fc4 = nn.Linear(128,32)
        self.fc5 = nn.Linear(32,4)

    def attention_net(self, lstm_output, final_state):

        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights

    def forward(self, x):
        x1, x2, x3 = x[0], x[1], x[2]
        # x0, x1, x2分别为第1,2,3句话。

        out1 = self.embedding(x1)  # [batch_size, seq_len, embeding]
        out1, (hs1, _) = self.lstm1(out1)
        out2 = self.embedding(x2)
        out2, (hs2, _) = self.lstm2(out2)  # 对三句话分别做LSTM，再将得到的向量拼接，再用LSTM进行特征提取，最后放入全连接网络分类。
        out3 = self.embedding(x3)
        out3, (hs3, _) = self.lstm1(out3)

        out_all = torch.cat([out1, out2, out3], dim=-1)
        out, (hs, _) = self.lstm4(out_all)

        att_out, attention = self.attention_net(out, hs)

        out = self.fc(att_out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class LSTM_att2(LSTM_att):

    def forward(self, x):
        x1, x2, x3 = x[0], x[1], x[2]
        # x0, x1, x2分别为第1,2,3句话。

        out1 = self.embedding(x1)  # [batch_size, seq_len, embeding]
        out1, (hs1, _) = self.lstm1(out1)
        out2 = self.embedding(x2)
        out2, (hs2, _) = self.lstm2(out2)  # 对三句话分别做LSTM，再将得到的向量拼接，再用LSTM进行特征提取，最后放入全连接网络分类。
        out3 = self.embedding(x3)
        out3, (hs3, _) = self.lstm1(out3)
        out_all = torch.cat([out1, out2, out3], dim=-1)
        hs_all = torch.cat([hs1,hs2,hs3],dim=-1)
        att_out, attention = self.attention_net(out_all, hs_all)
        out = self.fc3(att_out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.dropout(out)
        out = self.fc5(out)
        return out
