def F1(labels, predicts):

    TP_FN = TP_FP = TP =0
    for i in range(len(labels)):
        if labels[i] == predicts[i] and labels[i] != 'others':
            TP += 1
        if labels[i] != 'others':
            TP_FN += 1
        if predicts[i] != 'others':
            TP_FP += 1
    presion = TP/TP_FP
    recall = TP/TP_FN
    f1 = 2 * presion*recall/(presion + recall)
    return presion,recall,f1

def F1_sklearn(labels, predicts):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    #使用sklearn计算F1的方法
    metrics_new = {

        "precision": (lambda labels, predicts:  # 只对情感标签对应的类别 计算micro-precision
                      precision_score(labels, predicts, average='micro',
                                      labels=['happy', 'sad', 'angry'])),
        "recoll": (lambda labels, predicts:  # 只对情感标签对应的类别 计算micro-recall
                   recall_score(labels, predicts, average='micro',
                                labels=['happy', 'sad', 'angry'])),
        "f1": (lambda labels, predicts:  # 只对情感标签对应的类别 计算micro-f1-score
               f1_score(labels, predicts, average='micro',
                        labels=['happy', 'sad', 'angry'])),
    }
    out = []
    for title, metric in metrics_new.items():
        out.append(metric(labels, predicts))
    return out[0],out[1],out[2]


def get_F1(model_name='lstm_att'):

    labels = []
    with open('./data/dev.txt', 'r', encoding='utf-8') as fp:   #获得验证集上的标签
        next(fp)
        for content in fp.readlines():
            label = content.split('\t')[-1].strip()
            labels.append(label)

    predicts = []

    with open('./data/dev_predict_'+model_name+'.txt', 'r', encoding='utf-8') as fp:    #获得验证集上的预测结果
        for content in fp.readlines():
            predict = content.strip()
            predicts.append(predict)

    if len(labels) != len(predicts):
        print('error')
    else:
        #计算precision， recall， f1
        presion, recall, f1 = F1_sklearn(labels,predicts)
        print('presion is %.4f, recall is %.4f, f1 score is %.4f\n'%(presion,recall,f1))

if __name__ == '__main__':
    print('Attention based Bi-LSTM')
    get_F1('lstm_att2')

    print('Attention based Multilevel Bi-LSTM')
    get_F1()

    print('Multilevel Bi-LSTM')
    get_F1('lstm2')

    print('Single Bi-LSTM')
    get_F1('lstm1')


    print('bayes')
    get_F1('bayes')





