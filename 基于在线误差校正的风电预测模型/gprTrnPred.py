#函数功能：训练GPR模型
def gprTrnPred(Xtrn,Ytrn,testX,alpha_GPR):
    from sklearn.gaussian_process import GaussianProcessRegressor as GPR#表示高斯回归预测
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel,Matern,ExpSineSquared
    from Normal_XY.Standardization_XY import Standardization as Std

    "归一化"
    StdX = Std()
    StdY = Std()
    TrainX=StdX.normal(Xtrn)
    TestX=StdX.normal(testX)
    TrainY=StdY.normal(Ytrn)

    "模型训练"
    "模型训练"
    alpha = alpha_GPR##alpha就是添加到协方差矩阵对角线上的值
    n_restarts_optimizer = 10#规定了优化过程的次数
    #kernel = ConstantKernel(0.1, (0.01, 200)) * RBF(0.1, (0.01, 200))
    #kernel = 1 ** 2 + Matern(length_scale=100, nu=0.001)
    kernel=1**2*ExpSineSquared(length_scale=1,periodicity=3)
    # print("训练参数：\n",'alpha:',alpha,'\n','n_restarts_optimizer:',n_restarts_optimizer,'\n')
    model = GPR( n_restarts_optimizer=n_restarts_optimizer,kernel=kernel, alpha=alpha)#高斯回归预测，调用python中的GaussianProcessRegressor库函数
    model.fit(TrainX, TrainY)
    "预测"
    mu = model.predict(TestX)
    "反归一化"
    Prediction = StdY.renormal(mu)

    return Prediction