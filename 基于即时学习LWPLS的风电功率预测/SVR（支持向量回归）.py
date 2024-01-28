import torch.nn as nn
import torch
import numpy as np
from Normal_XY.Standardization_XY import Standardization
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.svm import SVR
import time

Start_time = time.time()

#预测参数
year=2012
lookback=30
sample_interval=3
feature_num=3

ahead_step=16
ahead_time=ahead_step*sample_interval*5
"导入训练集"
file_path='./A0Sample_creat/2011/按月'+'/'+str(ahead_step * sample_interval*5)+'mins/'+'Z_201102'+'_Samples_lookback='+str(lookback)+' ahead='+str(ahead_step * (sample_interval) * 5)+'mins '+'sample_interval=('+str(sample_interval*5)+')mins '+'feature_num='+str(feature_num)+'.npz'
# file='./'+'/A0Sample_creat/'+'/2011/'+'/全年/'+str(ahead_step * sample_interval *5)+'mins/'+'Z_'+year+'_Samples_lookback='+str(lookback)+' ahead='+str(ahead_step * (sample_interval) * 5)+'mins '+'sample_interval='+str(sample_interval*5)+'mins '+'feature_num='+str(feature_num)
Data_train=np.load(file_path, allow_pickle=True)
inputs_train=Data_train['inputsMonths']
outputs_train=Data_train['outputsMonths']
lookback_train=Data_train['lookback']
feature_num_train=Data_train['feature_num']
ahead_step_train=Data_train['ahead_step']

file='./A0Sample_creat/2012/按月'+'/'+str(ahead_step * sample_interval *5)+'mins/'+'Z_201202'+'_Samples_lookback='+str(lookback)+' ahead='+str(ahead_step * (sample_interval) * 5)+'mins '+'sample_interval=('+str(sample_interval*5)+')mins '+'feature_num='+str(feature_num)+'.npz'
Data_test=np.load(file, allow_pickle=True)
inputs_test=Data_test['inputsMonths']
outputs_test=Data_test['outputsMonths']

train_X,train_Y,test_X,test_Y=inputs_train,outputs_train,inputs_test,outputs_test
"数据归一化"
StaX=Standardization()
TrainX=StaX.normal(train_X)#返回归一化后的数据
TrainX=TrainX.reshape(-1,feature_num*lookback)

"输出归一化"
StaY=Standardization()
TrainY=StaY.normal(train_Y)#返回归一化后的数据
TrainY=TrainY.reshape(-1,1).ravel()

Stax=Standardization()
test_X=Stax.normal(test_X)
test_X = test_X.astype(float)
test_X=test_X.reshape(-1,lookback*feature_num_train)

svr=SVR(kernel='rbf',C=0.1)
svr.fit(TrainX,TrainY)
result1=svr.predict(test_X)
result1=result1.reshape(-1,1)
out=StaY.renormal(result1)
test_Y=test_Y.reshape(-1,1)
print(out)
print(test_Y)
print('RMSE:',np.sqrt(mean_squared_error(out,test_Y)))
print('R2:',r2_score(test_Y,out))