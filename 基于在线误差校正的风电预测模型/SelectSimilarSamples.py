def SelectSimilarSamples(trnX,trnY,testX,localNum,simiFunNumber,phai):
    "数据归一化"
    from Normal_XY import Standardization_XY
    import SimiFun
    import numpy as np
    StdX=Standardization_XY.Standardization()
    StdY=Standardization_XY.Standardization()

    TrnXsimi=StdX.normal(trnX)#归一化
    Xnewsimi=StdX.normal(testX)#归一化

    Xnewsimi=Xnewsimi.reshape(1,-1)

    #计算距离
    if simiFunNumber == 1:
        simi = SimiFun.simiFunEuclideanDistance(Xnewsimi, TrnXsimi, phai)
    elif simiFunNumber == 2:
        simi = SimiFun.simiFunCosine(Xnewsimi, TrnXsimi, phai)
    else:
        simi = SimiFun.simiFunMahalanobisDistance(Xnewsimi, TrnXsimi, phai)

    "获取最相似的前localNum个样本"
    hybridTrain = np.hstack((trnX, trnY, simi.T))#按水平方向（列顺序）拼接 2个或多个图像，图像的高度（数组的行）必须相同。
    hybridTrain = hybridTrain[hybridTrain[:, -1].argsort()[::-1]]
    # 提取目标样本localNum个最大相似样本
    simi = hybridTrain[:localNum, -1]
    simitrnX = hybridTrain[:localNum, :- 2]
    simitrnY = hybridTrain[:localNum, -2]
    return simitrnX,simitrnY,simi

