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
lookback=10
ahead=4
ahead_time=ahead*15
sample_interval=15
feature_num=3

local_num = 300## 提取目标样本localNum个最大相似样本
simiFunNumber = 1
phai = 1
alpha_GPR = 10

IFUp=False

"导入训练集"

Month=['02','03','04','05','06','07','08','09','10','11','12']
i=11#表示预测第i个月的数据
for month in [i]:
    month_Test=Month[month-2]#测试数据集，用第i个月的数据来预测误差值
    month_Trn=Month[month-3] #训练数据集，用第i-1月的数据来训练GPR模型

    file_Test='./A2ErrorSamplesAll/2012/按月/'+str(ahead_time)+'mins/2012'+month_Test+'REPSamples_lookback='+str(lookback)+'ahead='+str(ahead_time)+'sample_interval=15mins feature_num=3.npz'
    "导入测试数据"#测试数据集
    Data_Test = np.load(file_Test, allow_pickle=True)
    test_REP = Data_Test['REP']
    test_E_OUT = Data_Test['E_OUT']
    test_Predict_OUT = Data_Test['Predict_OUT'].reshape(-1, 1)
    test_Real_OUT = Data_Test['Real_OUT'].reshape(-1, 1)
    test_P_ahead = Data_Test['P_ahead']
    test_REP = test_REP.astype('float')

    "训练集"#训练数据集
    file_name_Trn='./A2ErrorSamplesAll/2012/按月/'+str(ahead_time)+'mins/2012'+month_Trn+'REPSamples_lookback='+str(lookback)+'ahead='+str(ahead_time)+'sample_interval=15mins feature_num=3.npz'
    Data_train=np.load(file_name_Trn,allow_pickle=True)
    train_REP = Data_train['REP']
    train_E_OUT = Data_train['E_OUT']
    train_Predict_OUT = Data_train['Predict_OUT'].reshape(-1, 1)
    train_Real_OUT = Data_train['Real_OUT'].reshape(-1, 1)
    train_P_ahead = Data_train['P_ahead']
    train_REP = train_REP.astype('float')

    ErrorPre=[]
    ErrorCorrectionResult=[]

    #每次返回test_REP,test_Real_OUT,test_E_OUT,test_Predict_OUT中的第i个数值
    for TestREPi,TestReali,TestEi,TestLSTMPrei,idx in zip(test_REP,test_Real_OUT,test_E_OUT,test_Predict_OUT,range(test_REP.shape[0])):
        "GPR真实值预测"
        TestREPi=TestREPi.reshape(1, -1)
        #trnX,trnY,testX,testY,localNum,ahead_step,simiFunNumber,phai,Updata

        #ErrorPrei为LWPS的真实预测值
        ErrorPrei = JITLGprTrnPred.JITLGprTrnPred(trnX=train_REP,#使用trnX来训练GPR模型
                                                trnY=train_E_OUT,#使用trnY来训练GPR模型
                                                 testX=TestREPi,
                                                 localNum=local_num,
                                                 simiFunNumber=simiFunNumber,
                                                 phai=phai,
                                                 alpha_GPR=alpha_GPR)

        "校正"
        ErrorPre.append(ErrorPrei)
        ErrorCorrectionResulti = TestLSTMPrei + ErrorPrei.reshape(1, -1)
        ErrorCorrectionResult.append(ErrorCorrectionResulti.item())
        print("第%0.0f/ %0.0f个样本！" % (idx + 1, len(test_REP)))


    ErrorPre=np.array(ErrorPre).reshape(-1,1)
    ErrorCorrectionResult=np.array(ErrorCorrectionResult).reshape(-1,1)
    "性能测试"
    R2_E = r2_score(test_E_OUT, ErrorPre)
    RMSE_E = np.sqrt(mean_squared_error(test_E_OUT, ErrorPre))

    R2_LSTMError = r2_score(test_Real_OUT, ErrorCorrectionResult)
    RMSE_LSTMError = np.sqrt(mean_squared_error(test_Real_OUT, ErrorCorrectionResult))

    "保存预测结果"
    file_save='./A3JITLResult/'+str(ahead_time)+'mins/'+'LSTMEcor 2012'+month_Test+'Ahead='+str(ahead_time)+'Lookback='+str(lookback)+'local_num='+str(local_num)+'phai='+str(phai)+'alpha='+str(alpha_GPR)+'SimiFun='+str(simiFunNumber)

    np.savez(file_save,
             testX=test_REP,
             testEY=test_E_OUT,
             testRealY=test_Real_OUT,
             Predict_Error=ErrorPre,
             Predict_LSTM=test_Predict_OUT,
             Predict_LSTMEcor=ErrorCorrectionResult
             )
    t=[t for t in range(len(ErrorCorrectionResult))]
    plt.plot(t,test_Predict_OUT)
    plt.plot(t,ErrorCorrectionResult)
    plt.plot(t,test_Real_OUT)
    plt.legend(['LSTM','LSTM+GPR','Real'])#创建图例，Real表示真实值，Predict表示预测值
    # plt.legend(['Real','Predict'])
    plt.xlabel("时间")
    plt.ylabel("功率")
    # plt.title('2012年'+str(i)+'月的JITL预测结果，'+'lookback='+str(lookback)+'，ahead_time='+str(ahead_time))
    plt.show()

    print('R2_LSTMError='+str(R2_LSTMError))
    print('RMSE_LSTMError='+str(RMSE_LSTMError))




