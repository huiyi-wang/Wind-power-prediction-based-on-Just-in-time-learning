import numpy as np
year=['2012']
months=['01','02','03','04','05','06','07','08','09','10','11','12']

lookback=5
ahead_step=1#（8*15=120mins=2h）小时预测
feature_num=3
ahead_time=ahead_step*15
sample_interval=3
for Y in year:
    for M in months:
        file_name='./'+Y+'/按月/'+'采样数据'+'/XY '+Y+M+'sample_interval='+str(sample_interval)+'('+str(sample_interval*5)+'mins).npz'

        Data=np.load(file_name,allow_pickle=True)
        #15分钟采样一次，一个间隔即为15分钟一次采样，则一步间隔：（sample_interval+1）*15 min
        X_Y=Data['XY']
        sample_interval=Data['sample_interval']

        "按月存储"
        inputsMonths = np.zeros((1,lookback,feature_num))
        outputsMonths = np.zeros((1,1))

        # "按天存储"
        # inputsDay = np.zeros((1,lookback,feature_num))  # 按天存储
        # outputsDay = np.zeros((1,1))

        for i in range(len(X_Y)-lookback-ahead_step):
            inputsMonths=np.append(inputsMonths,X_Y[i:i+lookback,:].reshape(1,lookback,feature_num),axis=0)
            outputsMonths=np.append(outputsMonths,X_Y[i+(lookback-1)+ahead_step,0])

        "保存月"
        inputsMonths=np.delete(inputsMonths,0,axis=0)
        outputsMonths=np.delete(outputsMonths,0,axis=0)

        np.savez('./'+Y+'/按月/'+str(ahead_step * sample_interval *5)+'mins/'+'Z_'+Y+M+'_Samples_lookback='+str(lookback)+' ahead='+str(ahead_step * (sample_interval) * 5)+'mins '+'sample_interval=('+str(sample_interval*5)+')mins '+'feature_num='+str(feature_num)+'',
                 inputsMonths=inputsMonths,
                 outputsMonths=outputsMonths,
                 lookback=lookback,
                 ahead_step=ahead_step,
                 feature_num=feature_num
                 )
        print(Y+M+"Lookback=%0.0f Ahead_time=%0.0f Sample Creat Finish！进度：%0.2f"%(lookback,ahead_time,100), "%")


# inputsDay = np.append(inputsDay, X_Y[i:i + lookback, :].reshape(1, lookback, feature_num), axis=0)
# outputsDay = np.append(outputsDay, X_Y[i + (lookback - 1) + ahead_step, 0])

# if (i%96==0 and i>0):#如果是96的倍数则说明是一天，存储
#     "保存数据"
#     save_path = './' + Y + '/按天/'+str(ahead_time)+'mins/Z_'
#     save_name=Y+M+str(int(i/96))+\
#               '_Samples_lookback=' + str(lookback)+ \
#               ' ahead=' + str(ahead_step * (sample_interval + 1) * 15) + \
#               'mins '+ 'sample_interval=' + str((sample_interval + 1) * 15) + \
#               'mins '+ 'feature_num=' + str(feature_num)
#     "删除第一行"
#     inputsDay = np.delete(inputsDay, 0, axis=0)
#     outputsDay = np.delete(outputsDay, 0, axis=0)
#
#     "存储"
#     np.savez(save_path+save_name,
#              inputsDay=inputsDay,
#              outputsDay=outputsDay,
#              lookback=lookback,
#              ahead_step=ahead_step,
#              feature_num=feature_num)
#     print(Y+M+str(int(i/96)))
#     "归零"
#     inputsDay = np.zeros((1, lookback, feature_num))
#     outputsDay = np.zeros((1,1))




