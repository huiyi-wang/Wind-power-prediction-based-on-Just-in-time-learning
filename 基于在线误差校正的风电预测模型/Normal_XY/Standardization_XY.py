class Standardization():
    def __init__(self,mean=None,std=None):
        self.mean=mean#平均值
        self.std=std  #标准偏差
    def caculat_mean_std(self,X):
        import numpy as np
        X = X.astype('float')
        if(self.mean is None):
            self.mean=np.mean(X,axis=0)
            self.std=np.std(X,axis=0)
        else:
            pass

    def normal(self,X):
        X=X.astype('float')
        self.caculat_mean_std(X=X)
        NormalX =(X-self.mean)/(self.std+0.0000001)
        return NormalX
    def renormal(self,NormalX):
        NormalX=NormalX
        RenormalX=NormalX*self.std+self.mean
        return RenormalX.astype('float')

    def print_mean_std(self):
        #print(self.mean,self.std)
        return self.mean,self.std
