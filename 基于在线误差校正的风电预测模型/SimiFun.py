#函数功能：计算出预测点与实际点之间的距离
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr


def simiFunEuclideanDistance(xi,Xj,phai):
    dxij=cdist(xi,Xj,'euclidean')#计算距离
    simi=np.exp(-dxij/(dxij.std(ddof=1)*phai))
    return simi

# metric : str or callable, optional
#         The distance metric to use.  If a string, the distance function can be
#         'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
#         'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
#         'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
#         'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
#         'wminkowski', 'yule'.
def simiFunCosine(xi,Xj,phai):
    dxij=cdist(xi,Xj,'cosine')#计算距离
    simi = np.exp(-dxij / (dxij.std(ddof=1) * phai))
    return simi

def simiFunMahalanobisDistance(xi, Xj, phai):
    dxij=cdist(xi,Xj,'mahalanobis')#计算距离
    simi = np.exp(-dxij / phai)
    return simi

# def simiFunPearsonr(xi, Xj, phai):
#     dxij=cdist(xi,Xj,'euclidean')
#     simi = np.exp(-dxij / phai)
#     return simi