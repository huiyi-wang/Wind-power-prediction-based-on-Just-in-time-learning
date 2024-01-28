class MixMinNormal():
    def __init__(self,Max=None,Min=None):
        self.Max=Max
        self.Min=Min
    def caculat_mean_std(self,X):
        X = X.astype('float')
        import numpy as np
        if(self.Max==None):
            self.Max=np.max(X,axis=0)
            self.Min=np.min(X,axis=0)
    def print_Max_Min(self):
        print("Max:",self.Max,"Min:",self.Min)
    def normal(self,X):
        self.caculat_mean_std(X=X)
        NormalX =(X-self.Min)/(self.Max-self.Min)
        return NormalX
    def renormal(self,NormalX):
        RenormalX=NormalX*(self.Max-self.Min)+self.Min
        return RenormalX