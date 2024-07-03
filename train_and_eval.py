import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from Loss import FocalLoss

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(num_epochs, model, train_iter, dev_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    lossFunction = FocalLoss(num_class=4,gamma=2)    # focal loss


    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        for i, batch in enumerate(train_iter):
            input = batch[0]
            labels = batch[1]
            outputs = model(input)
            model.zero_grad()
            loss = lossFunction(outputs, labels)   # focal loss

            # loss = F.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), './checkpoint.pth')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))

                model.train()
            total_batch += 1
            if total_batch - last_improve > 1000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        # scheduler.step()  # 学习率衰减

    test(model, dev_iter)


def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def test(model, dev_iter):
    # test
    model.load_state_dict(torch.load('./checkpoint.pth'))
    model.eval()

    test_acc, test_loss, test_report, test_confusion = evaluate(model, dev_iter, test=True)
    msg = 'Dev Loss: {0:>5.2},  Dev Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)



def predict(model, test_iter, filename='label.txt'):
    model.load_state_dict(torch.load('./checkpoint.pth'))
    model.eval()
    predict_all = np.array([], dtype=int)

    with torch.no_grad():
        count0 = count1 = count2 = count3 = 0
        for text, _ in test_iter:
            outputs = model(text)
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predict)

        with open('./data/' + filename, 'w', encoding='utf-8') as f:
            for predict in predict_all:
                if predict == 3:
                    label = 'angry'
                    count3 += 1
                if predict == 2:
                    label = 'sad'
                    count2 += 1
                if predict == 1:
                    label = 'happy'
                    count1 += 1
                if predict == 0:
                    label = 'others'
                    count0 += 1
                f.write(label + '\n')
            print('the number of others,happy,sad,angry is %d %d %d %d' % (count0, count1, count2, count3))