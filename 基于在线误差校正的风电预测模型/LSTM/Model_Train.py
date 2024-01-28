def train(Model,Config,inputs,outputs,Model_name):
    "输入输出转换为张量"
    import torch
    import numpy as np
    import time
    T1 = time.time()

    print(Config.model_structure+Model_name)

    inputs = inputs.astype(float)    # numpy强制类型转换
    outputs = outputs.astype(float)  # numpy强制类型转换
    inputs=torch.from_numpy(inputs)   # 张量转换
    outputs=torch.from_numpy(outputs) # 张量转换
    from torch.utils.data import DataLoader, TensorDataset
    Model.double()
    optimizer=Config.optim(Model.parameters(),lr=Config.learning_rate)
    loss_fun=torch.nn.MSELoss()

    train_loader=DataLoader(TensorDataset(inputs, outputs), batch_size=Config.batch_size)
    for epoch in range(Config.epoch):
        train_loss_array = []
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0], _data[1]
            optimizer.zero_grad()  # 训练前要将梯度信息置 0
            pred_y,hidden= Model(_train_X)  # 这里走的就是前向计算forward函数
            pred_Y=pred_y[:,-1,:].view(pred_y.shape[0],1,1)
            loss = loss_fun(pred_Y, _train_Y)  # 计算loss
            loss.backward()  # 将loss反向传播
            optimizer.step()  # 用优化器更新参数
            train_loss_array.append(loss.item())

        if (epoch%10==0):
            print('<LSTM_train>迭代次数为：{}/{}'.format(epoch,Config.epoch),'损失为：{}'.format(np.array(train_loss_array).mean()))

    torch.save(Model.state_dict(), Config.model_save_path + Config.model_structure+Model_name)
    T_train = time.time() - T1
    print("LSTM:"+Config.model_structure+"训练完毕,训练时长为:%.2f s" % T_train)
    print("LSTM模型训练结束，模型保存在:", Config.model_save_path, "路径下")