# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:59:40 2022

@author: DingJin
"""
#%%

'''basic setup'''

import finnhub
import json
import pandas as pd
import time, datetime

finnhub_client = finnhub.Client(api_key="c9dg38qad3id6u3ebjn0")

company_name = "AAPL"
start_dt = "2022-04-01"
end_dt = "2022-04-16"
resolution = "D"

#%%

fairplays = finnhub_client.company_peers(company_name)

#%%

'''
get stock news data
'''

def get_news_firm(company_lst, start_dt, end_dt):
    
    news_df = pd.DataFrame()
    
    for company in company_lst:
        
        comp_news_df = finnhub_client.company_news(company, _from=start_dt, to=end_dt)
        comp_news_df = pd.json_normalize(comp_news_df)
        
        if news_df.empty:
            news_df = comp_news_df
        else:
            news_df = pd.concat([news_df, comp_news_df],axis = 0)

    news_df["datetime"] = news_df["datetime"].map(lambda x:\
           datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
    return news_df 
        
news_firm = get_news_firm(fairplays, start_dt, end_dt)
news_firm.to_excel("news_firm.xlsx")
        
#%%
'''
get mkt news data
'''

def get_news_mkt():
    
    news_mkt = finnhub_client.general_news('general', min_id=0)
    news_mkt = pd.json_normalize(news_mkt)
    
    news_mkt["datetime"] = news_mkt["datetime"].map(lambda x:\
           datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
    
    return news_mkt
    
news_mkt = get_news_mkt()
news_mkt.to_excel("news_mkt.xlsx")

#%%
'''
get stock price data
'''
    
def get_price_firm(company_lst, start_dt, end_dt, resolution):
    
    start_unix = datetime.datetime.strptime(start_dt,"%Y-%m-%d")
    start_unix = int(time.mktime(start_unix.timetuple()))
    
    end_unix = datetime.datetime.strptime(end_dt,"%Y-%m-%d")
    end_unix = int(time.mktime(end_unix.timetuple()))
    
    price_df = pd.DataFrame()
    
    for company in company_lst:
        
        try:
            comp_price_df = finnhub_client.stock_candles(company, resolution, start_unix, end_unix)
        except:
            print("company "+company+" data not available!")
            continue
            
        comp_price_df = pd.json_normalize(comp_price_df)
        
        temp = pd.DataFrame(columns = comp_price_df.columns)
        i = 0
        
        data_len = len(comp_price_df.loc[0,"c"])
        for m in range(data_len):
            
            app_lst = []
            for col in comp_price_df.columns:
                if col!="s":
                    app_lst.append(comp_price_df.loc[0,col][m])
                else:
                    app_lst.append("ok")
            
            temp.loc[i] = app_lst
            i += 1
        
        comp_price_df = temp
        comp_price_df["company"] = company
        
        if price_df.empty:
            price_df = comp_price_df
        else:
            price_df = pd.concat([price_df, comp_price_df],axis = 0)
            
        
    return price_df


price_df = get_price_firm(fairplays, start_dt, end_dt, resolution)
price_df.to_excel("price_df.xlsx")

