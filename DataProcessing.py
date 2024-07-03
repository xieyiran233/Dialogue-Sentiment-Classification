# coding: UTF-8
import os
import torch
from tqdm import tqdm
import numpy as np
import re

# def remove_unchinese(file):   #去除文本中的非中文，经过测试发现效果变差
#     pattern = re.compile(r'[^\u4e00-\u9fa5]')
#     chinese = re.sub(pattern, '', file)
#     return chinese

def build_vocab(tokenizer, max_size, min_freq):

    # 构建一个词表：
    # 首先对数据集中的每一行句子按字进行分割，然后统计所有元素的出现频率
    # 接下来按照频率从高到低的顺序对所有频率大于min_freq的元素进行排序，取前max_size个元素
    # 最后按照频率降序构建字典vocab_dic：{元素: 序号}，vocab_dic的最后两个元素是 '<UNK>' 和 '<PAD>'
    vocab_dic = {}
    train_path = './data/train.txt'
    UNK, PAD = '<UNK>', '<PAD>'  # 分别表示未知字与padding

    with open(train_path, 'r', encoding='UTF-8') as f:
        next(f)
        for line in tqdm(f):
            lin = line.strip()
            if not lin:  # 跳过空行
                continue
            content = lin.split('\t')[0:3]
            for word in tokenizer(content[0]):  # 按字分割
                vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 统计第一列字频
            for word in tokenizer(content[1]):  # 按空格分割或者按字分割
                vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 统计第二列字频
            for word in tokenizer(content[2]):  # 按空格分割或者按字分割
                vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 统计第三列字频

        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

    return vocab_dic

def build_dataset():

    train_path = './data/train.txt'
    dev_path = './data/dev.txt'
    test_path = './data/test_without_label.txt'

    pad_size = 32
    MAX_VOCAB_SIZE = 10000  # 词表长度限制
    UNK, PAD = '<UNK>', '<PAD>'  # 分别表示未知字与padding

    tokenizer = lambda x: [y for y in x]
    vocab = build_vocab(tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    # print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=64):
        contents = []

        with open(path, 'r', encoding='UTF-8') as f:
            next(f)
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                try:
                    labels = lin.split('\t')[3].split('\n')[0]
                except:
                    labels = None
                if labels == 'others':
                    label = 0
                if labels == 'happy':
                    label = 1
                if labels == 'sad':
                    label = 2
                if labels == 'angry':
                    label = 3
                if labels is None:
                    label = -1


                content = lin.split('\t')[0]  # 获取数据
                words_line0 = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line0.append(vocab.get(word, vocab.get(UNK)))

                content = lin.split('\t')[1]  # 获取数据
                words_line1 = []
                token = tokenizer(content)
                seq_len = len(token)
                # 将语句转化为数字索引
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line1.append(vocab.get(word, vocab.get(UNK)))

                content = lin.split('\t')[2]  # 获取数据
                words_line2 = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line2.append(vocab.get(word, vocab.get(UNK)))

                contents.append((words_line0,words_line1,words_line2, int(label), seq_len))
                # words_line0,words_line1,words_line2分别为第1句，第2句，第3句话


        return contents

    train = load_dataset(train_path, pad_size)
    dev = load_dataset(dev_path, pad_size)
    test = load_dataset(test_path, pad_size)

    return vocab, train, dev, test

# vocab, train, dev, test = build_dataset()

class DatasetIterater(object):
    #  根据数据集产生batch

    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x0 = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        x1 = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        x2 = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # print(x0.size())
        # print(x1)
        # print(x2)
        y = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        return (x0, x1, x2, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, batch_size,device):

    iter = DatasetIterater(dataset, batch_size, device)

    return iter


if __name__ == "__main__":
    # 提取预训练词向量
    # 下面的目录、文件名按需更改。
    MAX_VOCAB_SIZE = 10000
    train_dir = "./train.txt"
    pretrain_dir = "./sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./embedding_SougouNews"


    tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    word_to_id = build_vocab(tokenizer = tokenizer,max_size=MAX_VOCAB_SIZE, min_freq=1)



    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        if i == 0:  # 若第一行是标题，则跳过
            continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)



