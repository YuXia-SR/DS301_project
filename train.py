import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime
import argparse
import json
import pickle

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

import sys
import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)
def get_train_data(train_config):
    df = pd.read_csv(train_config.train_file)
    df = df.set_index("Unnamed: 0")
    # process data: add covariance matrix
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
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
    return df
def experiment(train_config):
    if_nlp = train_config["IF_NLP"]
    ticker_list = train_config["TICKER_LIST"]
    start = train_config["START_DATE"]
    end = train_config["END_DATE"]
    train_test = train_config["TRAIN_TEST_SPLIT"]
    model_name = train_config["MODEL_NAME"]
    indicators = train_config["INDICATORS"]
    model_direct = train_config["MODEL_DIRECT"]
    data_direct = train_config["DATA_DIRECT"]
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
    df.to_csv(f'{data_direct}/{start}_{end}_nlp{if_nlp}.csv',index=False)
    cov_list = []
    lookback=train_config["LOOKBACK"] # look back is one year
    for i in range(lookback,len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        covs = return_lookback.cov().values 
        cov_list.append(covs)
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
    df = df.merge(df_cov, on='date')
    if if_nlp:
        nlp = pd.read_csv(f'{data_direct}/{train_config["NLP_FILE"]}')
        columns = ['date','tic']
        columns.extend(train_config["NLP_INDICATORS"])
        indicators.extend(train_config["NLP_INDICATORS"])
        nlp = nlp[columns]
        df = df.merge(nlp,on=["date","tic"],how='left')
        df.fillna(0,inplace=True)
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    # split data
    train = data_split(df, start, train_test)

    # prepare portfolioEnv
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    #print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
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
    e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    # initialize
    agent = DRLAgent(env = env_train)
    agent = DRLAgent(env = env_train)
    SAC_PARAMS = train_config["PARAMS"]

    model_sac = agent.get_model(model_name,model_kwargs = SAC_PARAMS)
    trained_sac = agent.train_model(model=model_sac, 
                                tb_log_name=model_name,
                                total_timesteps=train_config["T"])
    #trained_sac.save(
    #        f'{model_direct}/{model_name.upper()}_nlp{if_nlp}' # TODO how to save model in a better way
    #    )
    # create an iterator object with write permission - model.pkl
    #with open(f'{model_direct}/{model_name.upper()}_nlp{if_nlp}.pkl', 'wb') as files:
    #    pickle.dump(trained_sac, files)
    return trained_sac


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config",
                        help="Absolute path to configuration file.")
    args = parser.parse_args()

    # Ensure a config was passed to the script.
    if not args.config:
        print("No configuration file provided.")
        exit()
    else:
        with open(args.config, "r") as inp:
            config = json.load(inp)
    trained_model = experiment(config)
    with open(f'{config["MODEL_DIRECT"]}/{config["MODEL_NAME"].upper()}_nlp{config["IF_NLP"]}.pkl', 'wb') as files:
        pickle.dump(trained_model, files)