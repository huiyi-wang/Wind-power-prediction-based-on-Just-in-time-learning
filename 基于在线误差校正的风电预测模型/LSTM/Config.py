class Config:
    import torch.nn as nn
    import torch

    #%网络初始化参数
    input_size = 3    # LSTM输入大小
    hidden_size = 30  # LSTM隐层节点数（C(t)），即：一个LSTM单元有多少个神经元
    output_size = 1   # 网络输出大小（单输出）
    lstm_layers = 2   # LSTM堆叠层数，隐含层层数
    dropout_rate = 0.1  #一层的时候参数没用，遗忘率
    bidirectional=False

    # #数据构造参数
    # rate_of_train=0.8  #训练集比例是占所有样本比例
    # rate_of_test =0.1   #该测试集比例是占总样本数的比例
    # rate_of_valid=0.1   #验证集占总样本数的比例

    #训练参数配置
    optim=torch.optim.Adam#优化器
    loss_fun=nn.MSELoss#损失函数
    learning_rate = 0.01#学习率
    batch_size = 1000#批大小，一次训练的样本数目
    epoch=50#学习次数

    #模型保存配置
    model_save_path="../LSTM/trained_model/"
    model_save_path_Pre="./LSTM/trained_model/"
    model_structure="Hidden_size="+str(hidden_size)+"Lstm_layers="+str(lstm_layers)+"Learning_rate="+str(learning_rate)+"Epoch="+str(epoch)
