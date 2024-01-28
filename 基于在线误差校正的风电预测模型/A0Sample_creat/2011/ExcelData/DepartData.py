import pandas as pd
year=2011

filename = '89644-2011.csv'

Data = pd.read_csv(filename)
months=[1,2,3,4,5,6,7,8,9,10,11,12]
for M in months:
    Datam=Data[Data['Month']==M]
    file_save=str(M)+'月'+'.xlsx'
    Datam.to_excel(file_save)




# X_Y = Data.values[0::sample_interval, 5:]
# X_Y.reshape(-1,1)
# #print("Samping Finish!")
# return X_Y
#
# #months=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
# #M=['01','02','03','04','05','06','07','08','09','10','11','12']
# #sample_interval=3
# for month,m in zip(months,M):
#     filename = './2012/ExcelData/'+month+'.xlsx'
#     XY=Creat.Samping(filename,sample_interval)
#     np.savez('./'+str(year)+'/按月/采样数据/'+'XY '+str(year)+m+'sample_interval='+str(sample_interval)+"("+str((sample_interval)*5)+'mins)',XY=XY,sample_interval=sample_interval)
#     print(str(year)+'年'+month+'Samping finish')