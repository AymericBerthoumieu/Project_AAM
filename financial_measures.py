import statsmodels.api as sm
import numpy as np


def momentum(history: int, metric, take_last: bool = False):
    """
    :param history: depth of the momentum. e.g if 12 and data is monthly 12-months momentum computed
    :param metric: can be returns of the stocks, alpha or alpha + specific risk for example
    :param take_last: if true the most recent data is also taken into account
    :return:
    momentum computed using the last *history* values of the *metric*
    """
    if take_last:
        mom = metric.rolling(history).mean().dropna()
    else:
        mom = metric.rolling(history).apply(lambda x: np.mean(x[:-1])).dropna()
    return mom


def capm(returns: np.array, returns_market: np.array, rfr: float = 0):
    """
    :param returns: returns of the assets to regress
    :param returns_market : returns of the market
    :param rfr: risk free rate
    :return:
    alphas and betas for all assets
    """
    alpha_betas = np.zeros((2, returns.shape[1]))
    market = sm.add_constant(returns_market - rfr)
    for asset in range(returns.shape[1]):
        model = sm.OLS(returns[:, asset], market)
        results = model.fit()
        alpha_betas[:, asset] = results.params[1:]
    return alpha_betas
