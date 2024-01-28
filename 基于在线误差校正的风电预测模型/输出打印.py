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

file_Test = './A3JITLResult/240mins/LSTMEcor 201211Ahead=240Lookback=30local_num=200phai=1alpha=10SimiFun=1.npz'

Data_Test = np.load(file_Test, allow_pickle=True)

test_REP = Data_Test['testX']
test_E_OUT = Data_Test['testEY']
test_Real_OUT = Data_Test['testRealY']
ErrorPre = Data_Test['Predict_Error']
test_Predict_OUT = Data_Test['Predict_LSTM']
ErrorCorrectionResult = Data_Test['Predict_LSTMEcor']


t = [t for t in range(len(ErrorCorrectionResult))]
plt.plot(t, test_Predict_OUT)
plt.plot(t, ErrorCorrectionResult)
plt.plot(t, test_Real_OUT)
plt.legend(['LSTM', 'LSTM+GPR', 'Real'])  # 创建图例，Real表示真实值，Predict表示预测值
# plt.legend(['Real','Predict'])
plt.xlabel("时间")
plt.ylabel("功率")
# plt.title('2012年'+str(i)+'月的JITL预测结果，'+'lookback='+str(lookback)+'，ahead_time='+str(ahead_time))
plt.show()