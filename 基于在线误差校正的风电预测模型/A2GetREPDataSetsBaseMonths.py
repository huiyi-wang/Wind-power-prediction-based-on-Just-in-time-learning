#将LSTM预测出的数据和误差进行数据重构，构成一维数向量组
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
from LSTM import Config


"程序描述"
#从文件LSTMResult中读取每个月LSTM预测结果
#构造REP样本,r表示样本序列，e表示误差序列，p表示功率序列
#存储于ErrorSamplesAll文件内
#程序以循环的方式遍历每个月数据

ahead_time=120
lookback=12
year=2012
sample_interval=15
feature_num=3
months=['01','02','03','04','05','06','07','08','09','10','11','12']
#['01','02','03','04','05','06','07','08','09','10','11','12']

for month in months:
    #读取数据并赋值
    file_name='./A1LSTMResult/'+str(year)+'/按月/'+str(ahead_time)+"mins/"+str(year)+month+'LSTMResult_P_and_R'+"_lookback="+str(lookback)+"ahead="+str(ahead_time)+"sample_interval="+str(sample_interval)+"mins feature_num="+str(feature_num)+"epoch="+str(Config.Config.epoch)+".npz"
    Data=np.load(file_name,allow_pickle=True)
    inputs=Data['inputs']
    outputs=Data['outputs']
    TestX_LSTM=Data['TestX_LSTM']
    Predict_LSTM=Data['Predict_LSTM']
    Error_LSTM=Data['Error_LSTM']
    lookback=Data['lookback']
    ahead_step=Data['ahead_step']
    feature_num=Data['feature_num']

    "构建误差输入输出REP"
    Inputs = inputs.reshape(-1, lookback * feature_num)#由矩阵变为向量，张量转换

    #产生随机数组REP、E_OUT、Predict、Real_OUT、P_ahead，其中里面的数组全部为0

    REP=np.empty((1,lookback*feature_num+(lookback*2)))#分别是R(lookback长)，E(lookback长),P(lookback长)
    E_OUT=np.empty((1,1))#输出即误差项
    Predict_OUT=np.empty((1,1))
    Real_OUT=np.empty((1,1))
    P_ahead=np.empty((1,ahead_step))

    for i in range(len(Inputs)-(ahead_step+lookback-1)):#必须少移动lookback+ahead步，因为lookback+ahead步没有预测值
        E = Error_LSTM[i:lookback+i].reshape(1,-1)#误差项从0开始
        R=Inputs[(lookback+ahead_step)-1+i].reshape(1,-1)#真实值的第一个必须移动lookback+ahead步

        P=Predict_LSTM[i:lookback+i].reshape(1,-1)#预测值从0开始
        P_A=Predict_LSTM[lookback+i:lookback+ahead_step+i].reshape(1,-1)

        "堆叠在一起REP顺序"
        #np.hstack将参数元组的元素数组按水平方向进行叠加
        RE=np.hstack((R,E))
        REP1=np.hstack((RE,P))
        REP=np.append(REP,REP1,axis=0)
        E_OUT=np.append(E_OUT,Error_LSTM[(lookback+ahead_step)-1+i].reshape(1,1),axis=0)#误差的输出相当于重新构建时间序列，因此误差输出必须是lookback+ahead
        Predict_OUT=np.append(Predict_OUT,Predict_LSTM[(lookback+ahead_step)-1+i].reshape(1,1))
        Real_OUT=np.append(Real_OUT,outputs[(lookback+ahead_step)-1+i].reshape(1,1))
        P_ahead=np.append(P_ahead,P_A,axis=0)
        print(i+1)

    #删除0值
    REP=np.delete(REP,0,axis=0)
    E_OUT=np.delete(E_OUT,0,axis=0)
    Predict_OUT=np.delete(Predict_OUT,0,axis=0)
    Real_OUT=np.delete(Real_OUT,0,axis=0)
    P_ahead=np.delete(P_ahead,0,axis=0)
    np.savez('./A2ErrorSamplesAll/'+str(year)+'/按月/'+str(ahead_time)+"mins/"+str(year)+month+'REPSamples'+"_lookback="+str(lookback)+"ahead="+str(ahead_time)+"sample_interval="+str(sample_interval)+"mins feature_num="+str(feature_num),
             inputs=inputs,
             outputs=outputs,
             Predict_LSTM=Predict_LSTM,
             Error_LSTM=Error_LSTM,
             lookback=lookback,
             ahead_step=ahead_step,
             REP=REP,    #误差校正模型GPR模型的输入
             E_OUT=E_OUT,#误差校正模型GPR模型的输出
             Predict_OUT=Predict_OUT,
             Real_OUT=Real_OUT,
             P_ahead=P_ahead,
             feature_num=feature_num
             )
    print(str(year)+month+'ahead_time='+str(ahead_time)+"mins Lookback="+str(lookback)+"误差输入输出构造（REP）完成！")