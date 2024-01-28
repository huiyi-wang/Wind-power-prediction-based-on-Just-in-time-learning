#使用LSTM模型进行预测
import numpy as np
from Normal_XY import Standardization_XY
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
import Model_predict
from LSTM import Config
from sklearn.metrics import r2_score,mean_squared_error

"程序描述"
#通过使用2011年风电数据离线建立的LSTM预测每个月数据
#获得预测误差后存储于LSTMResult文件下
"文件配置"
ahead_time=240
lookback=30
sample_interval=15
feature_num=3
months=['01','02','03','04','05','06','07','08','09','10','11','12']#['01','02','03','04','05','06','07','08','09','10','11','12']

year=2012#此处的2012表示预测的年份
Train_num=12

for M in range((12-len(months)+1),13):
    month = months[M-(12-len(months)+1)]
    "读取训练归一化参数"
    "LSTM"
    # Model_name="Ahead="+str(ahead_time)+"mins"+"lookback="+str(lookback)
    Model_name="Ahead="+str(ahead_time)+"mins"+"lookback="+str(lookback)+"Train_num="+str(Train_num)
    #Paras_lstm表示归一化参数的路径名称
    Paras_lstm=np.load('./Normal_XY/Normal_Paras/LSTM_Train/Normal_Paras_'+Config.Config.model_structure+Model_name+'.npz')

    #x_mean表示inputs的平均值，x_std表示inputs的标准偏差
    #y_mean表示outputs的平均值，y_std表示outputs的标准偏差
    x_mean_lstm,x_std_lstm=Paras_lstm['x_mean'],Paras_lstm['x_std']
    y_mean_lstm,y_std_lstm=Paras_lstm['y_mean'],Paras_lstm['y_std']

    "读取原始样本"
    file_name='./A0Sample_creat/'+str(year)+'/按月/'+str(ahead_time)+'mins/Z_'+str(year)+month+'_Samples_lookback='+str(lookback)+' ahead='+str(ahead_time)+'mins sample_interval=('+str(sample_interval)+')mins feature_num='+str(feature_num)+'.npz'
    Data=np.load(file_name,allow_pickle=True)

    inputs=Data['inputsMonths']
    outputs=Data['outputsMonths']

    lookback=Data['lookback']
    ahead_step=Data['ahead_step']
    feature_num=Data['feature_num']

    "归一化"
    #类的实例化
    stdx_lstm=Standardization_XY.Standardization(x_mean_lstm,x_std_lstm)
    stdy_lstm=Standardization_XY.Standardization(y_mean_lstm,y_std_lstm)

    TestX_LSTM=stdx_lstm.normal(inputs.reshape(-1,lookback,feature_num))
    TestY=outputs

    "开始测试"
    "LSTM"
    Config.Config.ahead = ahead_step*15
    #利用LSTM模型计算预测值predictLSTM
    predictLSTM=Model_predict.predict(config=Config.Config,test_X=TestX_LSTM,Model_name=Model_name)
    "反归一化"
    Predict_LSTM=stdy_lstm.renormal(predictLSTM)#反归一化函数
    "计算误差"#做差求出误差值
    Error_LSTM=TestY.reshape(-1,1)-Predict_LSTM

    "性能测试"
    #R2表示线性回归决定系数，RMSE表示均方根误差
    R2=r2_score(TestY,Predict_LSTM)
    RMSE=np.sqrt(mean_squared_error(TestY,Predict_LSTM))
    t=[t for t in range(len(TestY))]
    plt.plot(t,TestY)
    plt.plot(t,Predict_LSTM)
    plt.legend(['Real','LSTM_Predict'])#创建图例，Real表示真实值，Predict表示预测值
    plt.xlabel("时间")
    plt.ylabel("功率")
    # plt.title(str(year)+'年第'+str(month)+'月的LSTM预测结果，'+'ahead_time='+str(ahead_time)+'，lookback='+str(lookback))
    plt.title('LSTM模型预测结果')
    plt.show()


    print('第'+str(M)+'月的预测情况',':')
    print('R2='+str(R2))
    print('RMSE='+str(RMSE))


    file_name2='./A1LSTMResult/'+str(year)+'/按月/'+str(ahead_time)+"mins/"+str(year)+month+'LSTMResult_P_and_R'+"_lookback="+str(lookback)+"ahead="+str(Config.Config.ahead)+"sample_interval="+str(sample_interval)+"mins feature_num="+str(feature_num)+"epoch="+str(Config.Config.epoch)+".npz"
    "保存预测结果"
    np.savez(file_name2,
             inputs=inputs,
             outputs=outputs,
             TestX_LSTM=TestX_LSTM,
             Predict_LSTM=Predict_LSTM,
             Error_LSTM=Error_LSTM,
             lookback=lookback,
             feature_num=feature_num,
             ahead_step=ahead_step)
    print(str(year)+month+'ahead_time='+str(ahead_time)+"Lookback="+str(lookback)+"预测完成！")



