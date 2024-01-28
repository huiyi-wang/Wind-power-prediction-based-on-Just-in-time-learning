#LSTM预测子函数
import torch
from torch.utils.data import DataLoader, TensorDataset
from LSTM import Define_net

#预测函数
def predict(config,test_X,Model_name):

    test_X = test_X.astype(float)#更改数据类型

    test_X=torch.from_numpy(test_X).float()#将数组转换为张量
    test_set=TensorDataset(test_X)         #用TensiorDataset来对test_X进行打包
    test_loader=DataLoader(test_set,batch_size=1)#负责数据的抽象

    #加载数据
    model=Define_net.Net(config)
    model.load_state_dict(torch.load(config.model_save_path_Pre+config.model_structure+Model_name))

    #定义Tensor保存预测结果
    result=torch.Tensor()

    #预测过程
    model.eval()
    result1,hidden=model(test_X)
    result=result1.detach().cpu().numpy()[:,-1,:]# 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据
    hidden=hidden[0].detach().cpu().numpy()
    return result

