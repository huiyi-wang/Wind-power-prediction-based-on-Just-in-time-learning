import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from LSTM import Config
from Normal_XY import Standardization_XY
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
from sklearn.metrics import r2_score,mean_squared_error
import JITLGprTrnPred

"程序描述"
#获取LWPLS预测值

"不同时长参数训练"
lookback=30
ahead=16
ahead_time=ahead*15
sample_interval=15
feature_num=3

local_num = 60## 提取目标样本localNum个最大相似样本
simiFunNumber = 1
phai = 1
alpha_GPR = 10
month_Test = [3]  # 测试数据集

IFUp=False

file_name = '../A3JITLResult/' + str(ahead_time) + 'mins/' + 'LSTMEcor 2012' + month_Test + 'Ahead=' + str(
    ahead_time) + 'Lookback=' + str(lookback) + 'local_num=' + str(local_num) + 'phai=' + str(phai) + 'alpha=' + str(
    alpha_GPR) + 'SimiFun=' + str(simiFunNumber)
