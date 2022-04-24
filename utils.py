import statsmodels.api as sm
import numpy as np


def volatility(history: int, returns, annualizing_factor):
    """
    :param history: depth of the volatility. e.g if 12 and data is monthly 12-months volatility computed
    :param returns: returns of the assets
    :param annualizing_factor: factor to convert the volatility into annual volatility
    :return:
    rolling volatility
    """
    return returns.rolling(history).std() * annualizing_factor


def momentum(history: int, metric, take_last: bool = False):
    """
    :param history: depth of the momentum. e.g if 12 and data is monthly 12-months momentum computed
    :param metric: can be returns of the stocks, alpha or alpha + specific risk for example
    :param take_last: if true the most recent data is also taken into account
    :return:
    momentum computed using the last *history* values of the *metric*
    """
    if take_last:
        mom = metric.rolling(history).mean()
    else:
        mom = metric.rolling(history).apply(lambda x: np.mean(x[:-1]))
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
    excess_market = sm.add_constant(returns_market - rfr)
    for asset in range(returns.shape[1]):
        model = sm.OLS(returns[:, asset] - rfr, excess_market)
        results = model.fit()
        alpha_betas[:, asset] = results.params
    return alpha_betas


def inverse_vol(long_short_position, volatilities):
    """
    :param long_short_position: pandas dataframe containing 1 for long positions and -1 for short positions
    :param volatilities: volatility of the assets
    :return:
    weights by asset based on the inverse of their volatility
    """
    # inverse volatilities
    cluster_weights = (long_short_position.T / volatilities[long_short_position.index]).to_frame("w").fillna(0)
    # rescaling
    # short positions
    cluster_weights[cluster_weights.w < 0] = cluster_weights[cluster_weights.w < 0] / abs(
        cluster_weights[cluster_weights.w <= 0]["w"].sum())
    # long position
    cluster_weights[cluster_weights.w > 0] = cluster_weights[cluster_weights.w > 0] / cluster_weights[
        cluster_weights.w > 0]["w"].sum()
    return cluster_weights


class Statistics:
    PERIOD_MDD = {
        "1 day": 1,
        "1 month": 21,
        "3 months": 63,
        "6 months": 126,
        "1 year": 252
    }

    def __init__(self, df, freq=12, rfr=0.025, CVaR_conf=0.95):
        self.df = df
        self.freq = freq
        self.rfr = rfr
        self.CVaR_confidence = CVaR_conf

        self.stats = {
            "performance": np.nan,
            "annualized return": np.nan,
            "annualized volatility": np.nan,
            "Sharpe ratio": np.nan,
            "maximum drawdown": {},
            f"CVaR ({round(self.CVaR_confidence * 100)}%)": np.nan
        }

    def run(self):
        self.returns()
        self.volatility()
        self.sharpe()
        self.maxdrawdown()
        self.CVaR()
        return self.stats

    def returns(self):
        lvl = self.df.add(1).cumprod()
        perf = lvl[-1] / lvl[0]
        years = self.df.shape[0] / self.freq
        ret = perf ** (1 / years) - 1

        self.stats["performance"] = round(perf * 100, 2)
        self.stats["annualized return"] = round(ret * 100, 2)

    def volatility(self):
        self.stats["annualized volatility"] = round(100 * self.df.std() * np.sqrt(self.df.shape[0] / self.freq), 2)

    def sharpe(self):
        self.stats["Sharpe ratio"] = (self.stats["annualized return"] - self.rfr * 100) / self.stats[
            "annualized volatility"]

    def maxdrawdown(self):
        lvl = self.df.add(1).cumprod().dropna()

        for period in self.PERIOD_MDD:
            if (252 / self.freq <= self.PERIOD_MDD[period]) and (
                    int(self.PERIOD_MDD[period] * self.freq / 252) < len(self.df)):
                off_set = int(self.PERIOD_MDD[period] * self.freq / 252) == 1
                roll_max = lvl.rolling(int(self.PERIOD_MDD[period] * self.freq / 252) + off_set, min_periods=1).max()
                daily_drawdown = lvl / roll_max - 1
                self.stats["maximum drawdown"][period] = round(min(daily_drawdown) * 100, 2)

        self.stats["maximum drawdown"]["full period"] = round(min(lvl / lvl.cummax() - 1) * 100, 2)

    def CVaR(self):
        self.stats[f"CVaR ({round(self.CVaR_confidence * 100)}%)"] = round(
            100 * self.df.quantile(1 - self.CVaR_confidence), 2)
