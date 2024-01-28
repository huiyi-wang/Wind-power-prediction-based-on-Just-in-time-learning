# def normal_Y(Y,y_mean=None,y_std=None):#归一化函数
#     import numpy as np
#     if y_mean is None:
#         y_mean=Y.mean()
#         y_std=Y.std()
#     Y_norma=(Y-y_mean)/y_std
#     return Y_norma,y_mean,y_std
#
# def normal_X(X,x_mean=None,x_std=None):
#     import numpy as np
#     X_num = X.shape[0]
#     X_feature=X.shape[2]
#     X=X.reshape(-1,X_feature)
#     if x_mean is None:
#         x_mean = np.mean(X, axis=0)
#         x_std = np.std(X)
#     X_norma = (X - x_mean) / x_std
#     X_norma=X_norma.reshape(X_num,-1,X_feature)
#     return X_norma, x_mean, x_std
#
# def normal_X_2(X,x_mean=None,x_std=None):
#     import numpy as np
#     X_num = X.shape[0]
#     X_feature=X.shape[1]
#     X=X.reshape(-1,X_feature)
#     if x_mean is None:
#         x_mean = np.mean(X, axis=0)
#         x_std = np.std(X)
#     X_norma = (X - x_mean) / x_std
#     X_norma=X_norma.reshape(X_num,X_feature)
#     return X_norma, x_mean, x_std

#训练集和测试集数据划分
def get_train_val_test_semple(inputs,outputs,rate_of_train,rate_of_test,rate_of_valid,if_random):
    #rate_of_train表示训练集占比
    #rate_of_test表示测试集占比
    #rate_of_valid表示交叉验证集占比
    import random   #随机函数
    import numpy as np
    inputs_num=len(inputs)#inputs的数目长度
    train_num=int(inputs_num*rate_of_train)#训练集的集合长度
    test_num=int(inputs_num*rate_of_test)  #测试集的集合长度
    val_num=int(inputs_num*rate_of_valid)  #交叉验证集集合长度

    if if_random:#打乱顺序
        random_index=random.sample(range(0,inputs_num),inputs_num)
    else:#不打乱顺序
        random_index=np.array(range(inputs_num)) #随机产生一个inputs.num长度的矩阵
    random_inputs=inputs[random_index]
    random_otputs=outputs[random_index]

    trainX=random_inputs[:train_num]
    trainY=random_otputs[:train_num]

    valX=random_inputs[train_num:train_num+val_num]
    valY=random_otputs[train_num:train_num+val_num]

    testX=random_inputs[train_num+val_num:]
    testY=random_otputs[train_num+val_num:]
    return trainX,trainY,valX,valY,testX,testY