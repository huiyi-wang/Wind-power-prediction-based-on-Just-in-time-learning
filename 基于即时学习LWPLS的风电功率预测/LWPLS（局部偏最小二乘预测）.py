#主函数
import numpy as np

from Normal_XY import Standardization_XY
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
from sklearn.metrics import r2_score,mean_squared_error
import LWPLSTrnPred

"程序描述"
#功能：通过验证集获得合适的参数
#参数包括：
#1.LWPLS:local_num,n_comp,phai,SimiFun

"不同时长参数训练"
lookback=30
ahead=16
ahead_time=ahead*15
sample_interval=15
feature_num=3

ParaSelectModel='JITL'#LSTM/LWPLS/JITLGPR

#LWPLS参数选择"#此部分训练集调用上一年的数据作为训练集

n_comps=[10]#n_comps表示建立局部模型时选定的潜变量个数

#local_nums=[500,800,1100,1500,2000]
phais=[1]#训练样本权值求解中的带宽参数
SimiFun=2#训练样本权值的编号

#选择第i月作为最小二乘LWPLS的预测月
Train='02'#用2011年第i月作为训练集
Test=Train#用2012年第i月作为验证集

for n_comp in n_comps:
    for phai in phais:
        "验证集"#使用2012年的数据为验证集
        file_name='./A0Sample_creat/'+str(2012)+'/按月/'+str(ahead_time)+'mins/Z_'+str(2012)+Test+'_Samples_lookback='+str(lookback)+' ahead='+str(ahead_time)+'mins sample_interval=('+str(sample_interval)+')mins feature_num='+str(feature_num)+'.npz'
        Data=np.load(file_name,allow_pickle=True)
        TestX_LWPLS=Data['inputsMonths'].reshape(-1,feature_num*lookback)#inputsMonths数据结构重建
        TestY_LWPLS=Data['outputsMonths'].reshape(-1,1)#outputsMonths数据结构重建

        "训练集"#使用2011年的数据为训练集
        file_name_Trn='./A0Sample_creat/'+str(2011)+'/按月/'+str(ahead_time)+'mins/Z_'+str(2011)+Train+'_Samples_lookback='+str(lookback)+' ahead='+str(ahead_time)+'mins sample_interval=('+str(sample_interval)+')mins feature_num='+str(feature_num)+'.npz'
        Data=np.load(file_name_Trn,allow_pickle=True)
        TrnX_LWPLS=Data['inputsMonths'].reshape(-1,feature_num*lookback)
        TrnY_LWPLS=Data['outputsMonths'].reshape(-1,1)

        local_num=TrnX_LWPLS.shape[0]#返回训练集合的矩阵长度

        Pre_LWPLS=[]#第一训练集预测结果

        for idx,testXi in zip(range(len(TestX_LWPLS)),TestX_LWPLS):#idx表示第i个数，testXi表示TestX_LWPLS的第i行数据
            testXi=testXi.reshape(1,-1)#reshape(1,-1)表示转化成1行
            Pre1i = LWPLSTrnPred.lwplsTrnPred(TrnX_LWPLS,TrnY_LWPLS,testXi,n_comp,local_num,phai,SimiFun)  # 第i个训练集预测结果
            #TrnX_LWPLS表示训练集的inputsMonths
            #TrnY_LWPLS表示训练集的outputsMonths
            #local_num表示训练集中inputsMonths的长度

            Pre_LWPLS.append(Pre1i)
            print("第%0.0f/ %0.0f个样本！" % (idx + 1, len(TestX_LWPLS)))

        Pre_LWPLS=np.array(Pre_LWPLS).reshape(-1,1)#第一训练集预测结果


"性能测试与比较"
#R2表示决定系数；R2 = 1表示样本中预测值和真实值完全相等，没有任何误差，表示回归分析中自变量对因变量的解释越好。
R2_LWPLS = r2_score(TestY_LWPLS,Pre_LWPLS)
# RMSE表示均方根误差，当预测值与真实值完全吻合时等于0，即完美模型；误差越大，该值越大
RMSE_LWPLS = np.sqrt(mean_squared_error(TestY_LWPLS,Pre_LWPLS))
"保存预测结果"
file_save = 'LWPLS/' +str(ahead_time)+'mins/'+ '2012'+Test+'LWPLS Ahead=' + str(ahead_time) + 'Lookback=' + str(
   lookback) +'n_comp='+str(n_comp)+'local_num='+str(local_num)+'phai='+str(phai)+'SimiFun='+str(SimiFun)+ 'R2=%0.4f RMSE=%0.4f' % (R2_LWPLS, RMSE_LWPLS)

np.savez(file_save,testX=TestX_LWPLS,testY=TestY_LWPLS,Predict_LWPLS=Pre_LWPLS)

print('LWPLS预测结束')

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

'输出性能'
print('R2_LWPLS'+'='+str(R2_LWPLS))
print('RMSE_LWPLS'+'='+str(RMSE_LWPLS))