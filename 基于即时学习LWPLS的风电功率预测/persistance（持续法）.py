import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
#预测参数
year=2012
lookback=30
sample_interval=3
feature_num=3

ahead_step=16
ahead_time=ahead_step*sample_interval*5
file='./A0Sample_creat/2012/按月'+'/'+str(ahead_step * sample_interval *5)+'mins/'+'Z_201202'+'_Samples_lookback='+str(lookback)+' ahead='+str(ahead_step * (sample_interval) * 5)+'mins '+'sample_interval=('+str(sample_interval*5)+')mins '+'feature_num='+str(feature_num)+'.npz'
Data_test=np.load(file, allow_pickle=True)
inputs_test=Data_test['inputsMonths']
outputs_test=Data_test['outputsMonths']

Pre_Persistence = outputs_test[:-ahead_step]
Real_Persistence = outputs_test[ahead_step:]

print("RMSE:",np.sqrt(mean_squared_error(Real_Persistence, Pre_Persistence)))
print("R2:",r2_score(Real_Persistence, Pre_Persistence))
