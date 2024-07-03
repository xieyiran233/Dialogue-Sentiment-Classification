# coding: UTF-8

import torch
import numpy as np
import LSTM
from train_and_eval import train, init_network,predict

from DataProcessing import build_dataset, build_iterator
from F1 import get_F1


if __name__ == '__main__':
    # 搜狗新闻:embedding_SougouNews.npz

    batch_size = 32
    device = 'cuda'
    epoch = 5
    # 选择模型，lstm1，lstm2，or lstm_att
    # lstm1是仅使用单个bi_lstm对最后一句话分析,lstm2是对三句话分别做LSTM，将得到的向量拼接，再用LSTM进行特征提取，最后放入全连接网络分类。
    # lstm_att是在lstm2的基础上加入了注意力机制。 lstm_att2在lstm_att的基础上部分简化。
    model_name = 'lstm2'
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset()
    train_iter = build_iterator(train_data, batch_size,device)

    dev_iter = build_iterator(dev_data, batch_size,device)
    test_iter = build_iterator(test_data, batch_size,device)

    if model_name == 'lstm1':
        model = LSTM.lstm1().to(device)
    elif model_name == 'lstm2':
        model = LSTM.lstm2().to(device)
    elif model_name == 'lstm_att':
        model = LSTM.LSTM_att().to(device)
    elif model_name == 'lstm_att2':
        model = LSTM.LSTM_att2().to(device)
    else:
        raise 'No such model'

    init_network(model)
    print(model.parameters)
    train(epoch ,model, train_iter, dev_iter)
    print('on dev dataset')
    predict(model, dev_iter,filename='dev_predict_'+model_name+'.txt')   # 得到模型在验证集上的分类结果
    get_F1(model_name)
    print('\non test dataset')
    predict(model, test_iter, filename='label.txt')     # 得到模型在测试集上的分类结果
