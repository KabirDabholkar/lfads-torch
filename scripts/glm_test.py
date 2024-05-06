import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from lfads_torch.metrics import bits_per_spike
import torch



class PrototypePoissonRegressor(PoissonRegressor):
    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csc", "csr"],
            dtype=[np.float64, np.float32],
            y_numeric=True,
            multi_output=False,
        )
        # required by losses
        if self.solver == "lbfgs":
            # lbfgs will force coef and therefore raw_prediction to be float64. The
            # base_loss needs y, X @ coef and sample_weight all of same dtype
            # (and contiguous).
            loss_dtype = np.float64
        else:
            loss_dtype = min(max(y.dtype, X.dtype), np.float64)
        
        self._base_loss = self._get_loss()

        Xzero = X[y==0].mean(axis=0)
        Xone  = X[y==1].mean(axis=0)
        # Xmean = X.mean(axis=0)
        diff = Xone - Xzero
        intercept = init_intercept = 0
        scaling = init_scaling = 0
        dot_prod = X @ diff
        learning_rate = 1e-3
        for i in range(self.max_iter):
            # print(scaling,intercept,scaling*dot_prod)
            dot_prod_intercept = dot_prod + intercept
            dscaling = np.mean(y * dot_prod_intercept - dot_prod_intercept * np.exp(scaling * dot_prod_intercept))
            dintercept = np.mean(scaling * (y-np.exp(scaling * dot_prod_intercept)))
            scaling += dscaling * learning_rate
            intercept += dintercept * learning_rate
            loglike = np.mean(y * scaling * dot_prod_intercept-np.exp(scaling * dot_prod_intercept))
            print(i,loglike)

        

        self.coef_ = diff * scaling
        self.intercept_ = intercept
        self.warm_start = True
        super().fit(X,y)
    


model = MultiOutputRegressor(estimator=PoissonRegressor())
# model = MultiOutputRegressor(estimator=PrototypePoissonRegressor())

np.random.seed(0)

num_neurons = 10
batch_size = 20
T = 100
y = np.sin(np.linspace(0,20,T*batch_size))
weights = np.random.normal(size=(num_neurons,)) * 0.1
X = weights[None] * y[:,None]
X = np.random.poisson(np.exp(X))
X = X.astype(np.int32)
X = X.reshape(batch_size,T,num_neurons)
X_train,X_test = np.split(X,[15])


n_samp = 8

model.fit(X_train.reshape(-1,X_train.shape[-1])[:n_samp],X_train.reshape(-1,X_train.shape[-1])[:n_samp])
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
