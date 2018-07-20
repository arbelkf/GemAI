import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import logging
style.use('ggplot')

start = dt.datetime(1995, 1, 1)
end = dt.datetime(2018, 7, 11)

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text,"lxml")
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open ("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers, f)
    print(tickers)
    return tickers


def take_first(array_like):
    return array_like[0]


def take_last(array_like):
    return array_like[-1]



def get_data_from_yahoo(reload_sp500=False,filepath = 'stock_dfs', daily=True):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open ("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists(filepath):
            os.makedirs(filepath)

    GetTickers(tickers, filepath, daily)

def GetTickers(tickers, filepath, daily):

    for ticker in tickers[:]:
        sys.stdout.write('.')
        if (daily == True):
            filename = filepath +'/{}.csv'.format(ticker)
        else:
            filename = filepath + '_weekly/{}.csv'.format(ticker)
        if not os.path.exists(filename):
            try:
                if (daily == True):
                    df = web.DataReader(ticker, 'yahoo', start , end)
                    df.to_csv(filename)
                else:
                    f = web.DataReader(ticker,'yahoo', start, end)
                    f.index = pd.to_datetime(f.index)

                    output = f.resample('W').agg({'Open': take_first,
                                                  'High': 'max',
                                                  'Low': 'min',
                                                  'Close': take_last,
                                                  'Adj Close': take_last,
                                                  'Volume': 'sum'})  # to put the labels to Monday

                    output = output[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                    output.index = output.index + pd.DateOffset(days=-6)
                    output.to_csv(filename)

                msg = "Getting {}".format(ticker)
                print(msg)
                #logging.Info(msg)
            except Exception as inst:
                msg = "cant find {} ".format(ticker) + "args:", inst.args[0]
                print(msg)
                logging.error(msg)
        #else:
            #print("Already have {}".format(ticker))


def GetCorraltionForSymbol(symbol):
    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()
    dfAns = df_corr[symbol].sort_values()
    print (dfAns)

def get_ndx_data_from_yahoo(reload_ndx=True, filepath = 'stock_dfs_ndx',daily=True):
    data = pd.read_csv("ndx.csv")
    tickers = data['ticker']

    if not os.path.exists(filepath):
            os.makedirs(filepath)

    GetTickers(tickers, filepath, daily)


get_ndx_data_from_yahoo(reload_ndx=True, filepath = 'stock_ndx',daily=True)
#get_data_from_yahoo(reload_sp500=False, filepath = 'stock_dfs', daily=True)
print("End")
