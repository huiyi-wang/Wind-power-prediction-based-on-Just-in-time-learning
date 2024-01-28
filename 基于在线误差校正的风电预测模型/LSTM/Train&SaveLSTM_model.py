#使用训练2011年的数据来训练LSTM模型
import numpy as np
from LSTM import Data_Process_Split
from LSTM import Define_net
from LSTM import Config
from LSTM import Model_Train
import matplotlib.pyplot as plt
from Normal_XY.Standardization_XY import Standardization
import time

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"读取2011年数据训练模型"#使用训练2011年的数据来训练LSTM模型
ahead_time=60 #15minxs（1步预测）、60mins（2步预测）、120mins（3步预测）、240mins（4步预测）
year=2011
lookback=10    #lookback=5、lookback=10、lookback=12、lookback=30
sample_interval=15
feature_num=3

Train_num=12

file_path='../A0Sample_creat/'+str(year)+'/全年/'+str(ahead_time)+'mins/Z_'+str(year)+'All_Samples_lookback='+str(lookback)+' ahead='+str(ahead_time)+'mins sample_interval='+str(sample_interval)+'mins feature_num='+str(feature_num)+'Train_num='+str(Train_num)+'.npz'
Data_train=np.load(file_path, allow_pickle=True)
inputs_train=Data_train['All_months_inputs']   #3个特征值的全部输入值，即inputs
outputs_train=Data_train['All_months_outputs'] #风电功率输出，即outputs
lookback_train=Data_train['lookback']          #
feature_num_train=Data_train['feature_num']    #特征值数目
ahead_step_train=Data_train['ahead_step']      #提前预测步骤

Lstm_Config=Config.Config
Lstm_Config.ahead=ahead_step_train*15
Lstm_Config.input_size=feature_num_train

#调用Data_Process_Split的子函数get_train_val_test_semple
train_X,train_Y,val_X,val_Y,test_X,test_Y=Data_Process_Split.get_train_val_test_semple(inputs_train,outputs_train,0.7,0.3,0.0,False)
#train_X表示inputs中的训练集矩阵
#train_Y表示outputs中的训练集矩阵
#test_X表示inputs中的测试集矩阵
#test_Y表示outputs中的测试集矩阵

"数据归一化"
StaX=Standardization()
TrainX=StaX.normal(train_X)#返回归一化后的数据
#TrainX为inputs中经过归一化的训练集矩阵

"输出归一化"
StaY=Standardization()
TrainY=StaY.normal(train_Y)#返回归一化后的数据
TrainY=TrainY.reshape(-1,1,1)
#TrainY为outputs中经过归一化的训练集矩阵

"实体化模型"
Model1=Define_net.Net(config=Lstm_Config)

Model_name="Ahead="+str(Lstm_Config.ahead)+"mins"+"lookback="+str(lookback_train)+'Train_num='+str(Train_num)
# Model_name="Ahead="+str(Lstm_Config.ahead)+"mins"+"lookback="+str(lookback_train)
'LSTM模型训练，并保存归一化参数'
'------------*-------*---------*--------*------------*--'
Start_time = time.time()
Model_Train.train(Model1,Lstm_Config,TrainX,TrainY,Model_name)
End_time = time.time()

print('LSTM训练时长为%f s'%(End_time-Start_time))

"归一化参数"
#将数据归一化并保存
#x_mean表示inputs的平均值，x_std表示inputs的标准偏差
#y_mean表示outputs的平均值，y_std表示outputs的标准偏差
x_mean,x_std=StaX.print_mean_std()#StaX=Standardization()表示数据归一化
y_mean,y_std=StaY.print_mean_std()#StaX=Standardization()表示数据归一化
np.savez('../Normal_XY/Normal_Paras/LSTM_Train/Normal_Paras_'+Config.Config.model_structure+Model_name,
         x_mean=x_mean,
         x_std=x_std,
         y_mean=y_mean,
         y_std=y_std)


