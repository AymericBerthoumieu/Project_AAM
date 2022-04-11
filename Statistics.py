import numpy as np


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
