import os
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error


ahead_time=str(240)+'mins/'

files= os.listdir('../LWPLS/'+ahead_time)
import Excel_Write_Result

for file,idx in zip(files,range(len(files))):
    if file[0]=='2':
        list = file.split('=')
        month = list[0][4:6]
        Ahead = list[1][:-8]
        lookback = list[2][:-6]
        n_comp = list[3][:-9]
        local_num = list[4][:-4]
        phai = list[5][:-7]
        SimiFun = list[6][:-2]
        # R2 = list[7][:-5]
        # RMSE = list[8][:-4]

        "读文件"
        Data=np.load('../LWPLS/'+ahead_time+file,allow_pickle=True)
        TestY_LWPLS=Data['testY']
        Pre_LWPLS=Data['Predict_LWPLS']

        "性能测试与比较"
        R2_LWPLS = r2_score(TestY_LWPLS, Pre_LWPLS)
        RMSE_LWPLS = np.sqrt(mean_squared_error(TestY_LWPLS, Pre_LWPLS))

        "写入Excel"
        result = [Ahead, month, lookback, n_comp, local_num, phai, SimiFun, R2_LWPLS, RMSE_LWPLS]
        Excel_Write_Result.Excle_Write('ZLWPLSParas.xls', result)
        print("第%0.0f/%0.0f组数据"%((idx+1),len(files)))
