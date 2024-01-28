#子函数
def JITLGprTrnPred(trnX,trnY,testX,localNum,simiFunNumber,phai,alpha_GPR):
    import SelectSimilarSamples
    import gprTrnPred
    import numpy as np
    print('GprTrnPred')
    Pre=[]
    testX=testX.astype('float')#数据类型转换
    trnX = trnX.astype('float')#数据类型转换trnX=train_REP
    trnY = trnY.astype('float')#数据类型转换trnY=train_E_OUT
    simiXtrn, simiYtrn, simi = SelectSimilarSamples.SelectSimilarSamples(trnX,trnY,testX,localNum,simiFunNumber,phai)
    pre=gprTrnPred.gprTrnPred(simiXtrn, simiYtrn.reshape(-1,1),testX,alpha_GPR=alpha_GPR)
    Pre.append(pre.item())#在列表pre最后(末尾)添加一个元素pre.item
    return np.array(Pre)