import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from lfads_torch.metrics import bits_per_spike
import torch
model = MultiOutputRegressor(estimator=PoissonRegressor())

np.random.seed(0)

num_neurons = 10
batch_size = 20
T = 100
y = np.sin(np.linspace(0,20,T*batch_size))
weights = np.random.normal(size=(num_neurons,))
X = weights[None] * y[:,None]
X = np.random.poisson(np.exp(X))
X = X.astype(np.int32)
X = X.reshape(batch_size,T,num_neurons)
X_train,X_test = np.split(X,[15])



model.fit(X_train.reshape(-1,X_train.shape[-1]),X_train.reshape(-1,X_train.shape[-1]))
Y_pred_test = model.predict(X_test.reshape(-1,X_test.shape[-1])).reshape(X_test.shape)
Y_pred_train = model.predict(X_train.reshape(-1,X_train.shape[-1])).reshape(X_train.shape)

co_bps_train = bits_per_spike(
    torch.tensor(np.log(Y_pred_train)),
    torch.tensor(X_train,dtype=torch.float32)
)

co_bps_test = bits_per_spike(
    torch.tensor(np.log(Y_pred_test)),
    torch.tensor(X_test,dtype=torch.float32)
)
#torch.mean(torch.from_numpy(Y.astype(np.float32))[None], dim=(0, 1), keepdim=True)
#print(model.n_features_in_,model.estimators_)

#print(torch.from_numpy(Y.astype(np.float32)))
#print(Y.dtype)
print('train co_bps',co_bps_train,'\ntest co_bps',co_bps_test)
