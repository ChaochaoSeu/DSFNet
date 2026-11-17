from baselines.SARIMA.arch import SARIMA


class SARIMAX(SARIMA):
    """y[t,n,k] = X[t,n] beta[n,:,k] + SARIMA(1,0,1)(0,1,1)[s] error.

    The MA polynomial uses (1-theta B)(1-Theta B**s). parameter_mode selects
    fixed phi/theta/Theta or train-only estimation; beta is always train-fitted.
    Channel indices refer to the original D dimension, before selection.
    """

    def __init__(self, *args, exog_features, **kwargs):
        if not exog_features:
            raise ValueError('SARIMAX needs at least one exogenous feature.')
        super().__init__(*args, exog_features=exog_features, **kwargs)
