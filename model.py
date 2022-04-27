# import packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime

from finrl import config
from finrl.config_tickers import DOW_30_TICKER
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
#from finrl.env.environment import EnvSetup
from finrl.finrl_meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
#from finrl.env.EnvMultipleStock_train import StockEnvTrain
#from finrl.env.EnvMultipleStock_trade import StockEnvTrade
from finrl.agents.elegantrl.models import DRLAgent
#from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot, backtest_strat, baseline_strat
#from finrl.trade.backtest import backtest_strat, baseline_strat

import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

# Download and save the data in a pandas DataFrame:
df = YahooDownloader(start_date = '2020-01-01',
                     end_date = '2020-12-01',
                     ticker_list = DOW_30_TICKER).fetch_data()

print(df.head())