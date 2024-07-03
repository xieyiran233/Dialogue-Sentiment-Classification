import os
import jieba
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def load_data(filename):
    labels = []
    data = []
    with open(filename, encoding='UTF8') as f:
        next(f)
        for contents in f.readlines():
            data.append(contents.split('\t')[2])
            label = contents.split('\t')[-1].strip()
            if label == 'others':
                labels.append(0)
            if label == 'happy':
                labels.append(1)
            if label == 'sad':
                labels.append(2)
            if label == 'angry':
                labels.append(3)
    return data, labels


def separateWords(Data):
    stopwords = [word.strip() for word in open('stop.txt', 'r', encoding='utf-8').readlines()]
    seqData = []
    all_words = []
    for text in Data:
        words = []
        seg_list = jieba.cut(text)
        for seg in seg_list:
            if (seg.isalpha() and seg != '\n' and seg not in stopwords):
                words.append(seg)
                all_words.append(seg)
        sentence = " ".join(words)
        seqData.append(sentence)

    return seqData


train, trainlabel = load_data("./data/train.txt")
dev, testlabel = load_data("./data/dev.txt")

train = separateWords(train)
dev = separateWords(dev)

vec = CountVectorizer()
traintxt = vec.fit_transform(train)
dev = vec.transform(dev)

mnb = MultinomialNB(alpha=0.3)
mnb.fit(traintxt, trainlabel)
predict = mnb.predict(dev)
predicts = []

for i in predict:
    if int(i) == 0:
        predicts.append('others')
    if int(i) == 1:
        predicts.append('happy')
    if int(i) == 2:
        predicts.append('sad')
    if int(i) == 3:
        predicts.append('angry')

with open('./data/dev_predict_bayes.txt', 'w', encoding='utf-8') as f:
    for content in predicts:
        f.write(content + '\n')

print(classification_report(testlabel, predict))
