# Dialogue-Sentiment-Classification
对话情感分类
## 任务描述
输入两人三段式(A-B-A)对话，预测第三轮对话的情感极性，其中情感极性包括 happy、sad、angry、others 四类。
## 方法
使用贝叶斯方法实现:
首先使用jieba对文本进行分词,然后计算词频矩阵,随后调用贝叶斯分类器进行分类。

使用LSTM方法实现:
首先使用两个Bi-LSTM提取说话人A和说话人B三句文本的语句级别特征,然后将得到的embedding特征拼接，然后使用 Bi-LSTM提取上下文特征,最后使用全连接分类。

使用BERT方法实现:
首先使用BERT对说话人A和说话人B的三句文本进行编码,然后将得到的embedding特征拼接,然后使用 Bi-LSTM提取上下文特征,最后使用全连接分类。
