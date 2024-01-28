def Samping(file_name,sample_interval):
    #input：文件名（Excel表），采样间隔
    #output:返回经过采样后的数据，包括输入输出
    import numpy as np
    import pandas as pd
    Data = pd.read_excel(file_name)

    X_Y = Data.values[0::sample_interval, 6:9]
    X_Y.reshape(-1,1)
    #print("Samping Finish!")
    return X_Y