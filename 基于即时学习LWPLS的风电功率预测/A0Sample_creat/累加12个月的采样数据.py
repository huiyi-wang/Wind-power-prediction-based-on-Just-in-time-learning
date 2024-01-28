import numpy as np

Year=['2011']
Months=['01','02','03','04','05','06','07','08','09','10','11','12']
#['01','02','03','04','05','06','07','08','09','10','11','12']
ahead_time=240
lookback=30
feature_num=3
sample_interval=3
for year in Year:
    # lookback = 0
    # ahead_step = 0
    # feature_num = 0
    All_months_inputs=np.zeros((1,lookback,feature_num))
    All_months_outputs = np.zeros((1,1))

    file_name2='../A0Sample_creat/'+year+'/全年/'+str(ahead_time)+'mins/'+'Z_'+year+'All_Samples_lookback='+str(lookback)+' ahead='+str(ahead_time)+'mins sample_interval='+str(sample_interval*5)+'mins feature_num='+str(feature_num)

    for month in Months:
        file_name1='../A0Sample_creat/'+year+'/'+'按月'+'/'+str(ahead_time)+'mins/Z_'+year+month+'_Samples_lookback='+str(lookback)+' ahead='+str(ahead_time)+'mins sample_interval=('+str(sample_interval*5)+')mins feature_num='+str(feature_num)+'.npz'
        Data=np.load(file_name1,allow_pickle=True)
        inputsMonths =Data['inputsMonths']
        outputsMonths =Data ['outputsMonths'].reshape(-1,1)
        lookback =Data ['lookback']
        ahead_step =Data ['ahead_step']
        feature_num =Data ['feature_num']
        "按月累叠在一起"
        All_months_inputs=np.append(All_months_inputs,inputsMonths,axis=0)
        All_months_outputs=np.append(All_months_outputs,outputsMonths,axis=0)

        print(month+'Added!')
        "保存数据"
    All_months_inputs=np.delete(All_months_inputs,0,axis=0)
    All_months_outputs=np.delete(All_months_outputs,0,axis=0)

    All_months_inputs.astype(float)
    All_months_outputs.astype(float)

    np.savez(file_name2,
             All_months_inputs=All_months_inputs,
             All_months_outputs=All_months_outputs,
             lookback=lookback,
             ahead_step=ahead_step,
             feature_num=feature_num
             )
    print('<Combine_All_Months>'+year+" Lookback="+str(lookback)+" Ahead_time="+str(ahead_time)+'完成合并！')
print(str(year)+'年完成数据合并')
