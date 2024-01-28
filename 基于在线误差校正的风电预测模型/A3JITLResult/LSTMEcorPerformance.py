import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
from sklearn.metrics import r2_score,mean_squared_error
import Excel_Write_Result

"不同时长参数训练"
lookback=10
ahead=4
ahead_time=ahead*15
sample_interval=15
feature_num=3

local_num = 60
simiFunNumber = 1
phai = 1
alpha_GPR = 10

months=['03','04','05','06','07','08','09','10','11','12']

for month in [3]:#,4,5,6,7,8,9,10,11,12
    month_Test=months[month-3]
    file='./'+str(ahead_time)+'mins/'+'LSTMEcor 2012'+month_Test+'Ahead='+str(ahead_time)+'Lookback='+str(lookback)+'local_num='+str(local_num)+'phai='+str(phai)+'alpha='+str(alpha_GPR)+'SimiFun='+str(simiFunNumber)+'.npz'
    Data=np.load(file, allow_pickle=True)

    testX = Data['testX']
    testEY = Data['testEY']
    testRealY =Data ['testRealY']
    Predict_Error =Data ['Predict_Error']
    Predict_LSTM =Data ['Predict_LSTM']
    Predict_LSTMEcor =Data ['Predict_LSTMEcor']


    Error_LSTM=testRealY.rehape()-Predict_LSTM

    "性能测试"
    R2_E = r2_score(testEY, Predict_Error)
    RMSE_E = np.sqrt(mean_squared_error(testEY, Predict_Error))

    R2_LSTMError = r2_score(testRealY, Predict_LSTMEcor)
    RMSE_LSTMError = np.sqrt(mean_squared_error(testRealY, Predict_LSTMEcor))

    R2_LSTM = r2_score(testRealY, Predict_LSTM)
    RMSE_LSTM = np.sqrt(mean_squared_error(testRealY, Predict_LSTM))

    Up=(RMSE_LSTM-RMSE_LSTMError)/RMSE_LSTM*100

    "画图"
    t=[t for t in range(len(testRealY))]
    plt.plot(t,testRealY)
    plt.plot(t,Predict_LSTM)
    plt.plot(t,Predict_LSTMEcor)
    plt.legend(['Real','LSTM','LSTMEcor'])
    plt.title('LWPLS:Ahead:%0.0f month:%s'%(ahead_time,month_Test))

    result=[ahead_time,month_Test,lookback,local_num,phai,alpha_GPR,simiFunNumber,R2_E,R2_LSTM,R2_LSTMError,RMSE_E,RMSE_LSTM,RMSE_LSTMError,Up]
    #Excel_Write_Result.Excle_Write('ZLSTMEcorResults.xls', result)
    print(month_Test)