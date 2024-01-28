import A0Sample_creat.Samping as Creat
import numpy as np

year=2011
months=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
M=['01','02','03','04','05','06','07','08','09','10','11','12']
sample_interval=3
for month,m in zip(months,M):
    filename = './'+str(year)+'/ExcelData/'+month+'.xlsx'
    XY=Creat.Samping(filename,sample_interval)
    np.savez('./'+str(year)+'/按月/采样数据/'+'XY '+str(year)+m+'sample_interval='+str(sample_interval)+"("+str((sample_interval)*5)+'mins)',XY=XY,sample_interval=sample_interval)
    print(str(year)+'年'+month+'Samping finish')

print(str(year)+'年采样完成')