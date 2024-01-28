    #References：
    # [1] Kim, S.; Kano, M.; Hasebe, S.; Takinami, A.; Seki, T. Long-Term Industrial
    #  Applications of Inferential Control Based on Just-In-Time Soft-Sensors:
    # Economical Impact and Challenges. Industrial & Engineering Chemistry Research 2013, 52, 12346-12356.
    # [2] Schaal, S.; Atkeson, C. G.; Vijayakumar, S. Scalable techniques from
    # nonparametric statistics for real time robot learning. Applied Intelligence 2002, 17, 49-60.
    # [3]Chen, M.; Khare, S.; Huang, B. A unified recursive just-in-time approach with
    # industrial near infrared spectroscopy application. Chemometrics and Intelligent Laboratory Systems 2014, 135, 133-140.
    # LWPLS主要参考文献[1][2]编写，文中把各变量的行列交代得比较清楚；文献[3]方法介绍清楚，但有部分表达错误。该论文中提取tr的计算有误，应该是提取（Xr'*W*Yr）*（Xr'*W*Yr）’最大特征值对应的特征向量
    # 文献[3]所述方法用于比较研究。
#子函数
def lwplsTrnPred(trnX,trnY,xnew,ncomp,localNum,phai,simiFunNumber):
    # TrnX_LWPLS,TrnY_LWPLS,testXi,n_comp,local_num,phai,SimiFun

    # TrnX_LWPLS=trnX表示训练集的inputsMonths
    # TrnY_LWPLS=trnY表示训练集的outputsMonths
    # testXi=xnew表示TestX_LWPLS的第i行数据
    # local_num=ncomp表示训练集中inputsMonths的长度

    from Normal_XY import Standardization_XY
    import SimiFun
    import numpy as np

    "归一化"
    StdX=Standardization_XY.Standardization()
    StdY=Standardization_XY.Standardization()
    trnX=StdX.normal(trnX)
    trnY=StdY.normal(trnY)
    xnew=StdX.normal(xnew)

    trnX = trnX.astype('float')#数据类型转换
    trnY = trnY.astype('float')#数据类型转换
    xnew = xnew.astype('float')#数据类型转换

    print('LWPLSTrnPred已被调用')


    if simiFunNumber==1:#选择计算距离的方法，此处选择的是SimiFun=simiFunNumber=1
        simi=SimiFun.simiFunEuclideanDistance(xnew, trnX, phai)
    elif simiFunNumber==2:
        simi=SimiFun.simiFunCosine(xnew, trnX, phai)
    else:
        simi=SimiFun.simiFunMahalanobisDistance(xnew, trnX, phai)
    #Select localNum most similar samples for local modeling
    hybridTrain =np.hstack((trnX, trnY, simi.T))#沿着水平方向将数组堆叠起来
    hybridTrain=hybridTrain[hybridTrain[:,-1].argsort()[::-1]]


    #提取目标样本localNum个最大相似样本
    simi = hybridTrain[:localNum, -1]
    simitrnX = hybridTrain[:localNum, :- 2]
    simitrnY = hybridTrain[:localNum, -2]
    X = simitrnX
    Y = simitrnY.reshape(-1,1)

    W=np.diag(simi)#构建权系数对角矩阵
    xnew = xnew.T
    xmu = ((W .dot(X)).sum(axis=0) / simi.sum(axis=0)).reshape(1,-1)#均值化
    ymu = ((W .dot(Y)).sum(axis=0) / simi.sum(axis=0)).reshape(1,-1)#均值化
    Xr = X-xmu
    Yr = Y-ymu
    xnewr = xnew - xmu.T
    ypred = ymu
    #Calculate the weights vector, loading vector and score vector

    #r为迭代步数，迭代计算
    for r in range(1,ncomp+1):
        ur = (Xr.T.dot(W).dot(Yr)).reshape(-1,1)
        wr=ur.dot(ur.T)
        #wr=wr.astype('float64')
        #eigvalue,eigvector= np.linalg.eig(wr)

        #print('空值',np.isnan(wr))
        #print('无穷',np.isinf(wr))

        lamda = np.linalg.eig(wr)
        index = np.argmax(lamda[0])
        lamda_max = np.real(lamda[0][index])
        vector = lamda[1][:, index]

        vector_final = np.transpose((np.real(vector)))
        ur = (vector_final).reshape(-1,1)
        tr = Xr .dot(ur)
        pr = Xr.T.dot(W).dot(tr)/tr.T.dot(W).dot(tr)
        qr =Yr.T.dot(W).dot(tr)/tr.T.dot(W).dot(tr)
        tqr = xnewr.T.dot(ur)

        #Deflate the input and output matrices
        Xr = Xr - tr.dot(pr.T)
        Yr = Yr - tr .dot(qr.T)
        xnewr = xnewr - tqr*pr

        #lwpls prediction
        ypred = ypred + tqr .dot( qr)
   #预测输出反归一化
    ypred = ypred
    ypred = StdY.renormal(ypred)
    return ypred