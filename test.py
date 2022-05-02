import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime
import argparse
import json

import pyfolio
from pyfolio import timeseries
import finrl.config as config
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer,data_split
from finrl.finrl_meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
#from finrl.preprocessing.data import data_split
#from finrl.env.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
#from finrl.model.models import DRLAgent
#from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
#import train_config

def test_model(train_config):
    add_nlp = train_config["ADD_NLP"]
    ticker_list = train_config["TICKER_LIST"]
    start = train_config["START_DATE"]
    end = train_config["END_DATE"]
    train_test = train_config["TRAIN_TEST_SPLIT"]
    model_name = train_config["MODEL_NAME"]
    indicators = train_config["INDICATORS"]
    model_direct = train_config["MODEL_DIRECT"]
    #data_direct = train_config["DATA_DIRECT"]

    # Download data
    df = YahooDownloader(start_date = start,
                        end_date = end,
                        ticker_list = ticker_list).fetch_data()
    # process data: add tech_indicators
    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        use_turbulence=False,
                        user_defined_feature = False)
    df = fe.preprocess_data(df)
    # process data: add covariance matrix
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
    df.to_csv(f'datasets/{start}_{end}.csv',index=False)
    cov_list = []
    lookback=252 # look back is one year
    for i in range(lookback,len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        covs = return_lookback.cov().values 
        cov_list.append(covs)
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    # split data
    trade = data_split(df, train_test, end)

    # prepare portfolioEnv
    stock_dimension = len(trade.tic.unique())
    state_space = stock_dimension
    env_kwargs = {
        "hmax": train_config["HMAX"], 
        "initial_amount": train_config["INITIAL_AMOUNT"], 
        "transaction_cost_pct": train_config["TRANSACTION_COST"], 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": indicators, 
        "action_space": stock_dimension, 
        "reward_scaling": train_config["REWARD_SCALING"]
        
    }
    trained_model # TODO define ways to load trained_model
    e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_model,
                        environment = e_trade_gym)
    DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func( returns=DRL_strat, 
                                factor_returns=DRL_strat, 
                                    positions=None, transactions=None, turnover_denom="AGB")
    baseline_df = get_baseline(
        ticker=train_config["BASE_TICK"], start=start, end=end
    )
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(returns = DRL_strat,
                                       benchmark_rets=baseline_returns, set_context=False)