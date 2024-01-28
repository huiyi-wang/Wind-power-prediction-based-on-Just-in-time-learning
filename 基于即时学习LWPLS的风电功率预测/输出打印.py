import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
from Normal_XY import Standardization_XY
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
from sklearn.metrics import r2_score,mean_squared_error


file_Test = './LWPLS/240mins/201202LWPLS Ahead=240Lookback=30n_comp=30local_num=2642phai=1SimiFun=1R2=0.5590 RMSE=3.9865.npz'

Data_Test = np.load(file_Test, allow_pickle=True)

TestX_LWPLS = Data_Test['testX']
TestY_LWPLS= Data_Test['testY']
Pre_LWPLS = Data_Test['Predict_LWPLS']

'画图'
t = [t for t in range(len(TestY_LWPLS))]
plt.plot(t, TestY_LWPLS)
plt.plot(t, Pre_LWPLS)
plt.legend(['Real', 'Predict'])  # 创建图例，Real表示真实值，Predict表示预测值
plt.xlabel('时间')
plt.ylabel('功率')
plt.title('LWPLS预测结果')
# plt.title('2012年第'+Train+'个月'+str(ahead_time)+'mins'+'预测结果，'+'潜变量个数为'+str(n_comps)+',lookback='+str(lookback))
plt.show()
