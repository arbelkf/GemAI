# Auther : Kfir Arbel
# GemAI Class


from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader as web
import stockstats
from matplotlib import style
style.use('ggplot')


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import Utils.Common as common
import logging
from StrategyModule.AbstractStrategy import AbstractStrategy

from StrategyModule.PercentForPeriodStrategy import PercentForPeriodStrategy
from StrategyModule.Context import Context




def do_ml(ticker):
    #extract startegy and class
    for strategy in strategyList:
        context = Context.Context(strategy)
        context.ProcessData(ticker)


def reloadALL():
    index = 0
    skipArray = ['BRK.B','BF.B']#,'IPGP','UNM','VFC','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX', 'VIAB']
    with open ("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    for ticker in tickers[:]:
        try:
            index += 1
            #if not os.path.exists(common.filePathProcessedStocks +'\processed_{}.csv'.format(ticker)):
            if (ticker not in skipArray) :
                msg = index, ": processing {}   ".format(ticker), index
                print(msg)
                logging.info(msg)
                do_ml(ticker)
            #print("skipping {}".format(ticker))

            #else:
            #    print(index, ": skipping {}   ".format(ticker), index)

        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])

            logging.error(type(inst))  # the exception instance
            logging.error(inst.args)  # arguments stored in .args
            logging.error(inst)  # __str__ allows args to be printed directly,
            logging.error("args:", inst.args[0])





strategyList = []
strategy = PercentForPeriodStrategy.PercentForPeriodStrategy(isIndicatorLongList=False,
                                                             filePathProcessedStocks="ProcessedStocks",
                                                             highestLimit=0.02, lowestLimit=0.02, hm_days=7, name ="ProcessedStocks",isDaily=True)
strategyList.append(strategy)
strategy = PercentForPeriodStrategy.PercentForPeriodStrategy(isIndicatorLongList = False, filePathProcessedStocks = "ProcessedStocksHighLimitLongHM", highestLimit =  0.04, lowestLimit = 0.04, hm_days = 14, name="ProcessedStocksHighLimitLongHM")
strategyList.append(strategy)
strategy = PercentForPeriodStrategy.PercentForPeriodStrategy(isIndicatorLongList = True, filePathProcessedStocks = "ProcessedStocksLongIndicators", highestLimit =  0.02, lowestLimit = 0.02, hm_days = 7, name="ProcessedStocksLongIndicators",isDaily=False)
strategyList.append(strategy)
strategy = PercentForPeriodStrategy.PercentForPeriodStrategy(isIndicatorLongList=True,
                                                                 filePathProcessedStocks="ProcessedStocksLongIndicatorsHighLimitLongHM",
                                                                 highestLimit=0.04, lowestLimit=0.04, hm_days=14, name = "ProcessedStocksLongIndicatorsHighLimitLongHM",isDaily=False)
strategy = PercentForPeriodStrategy.PercentForPeriodStrategy(isIndicatorLongList = False, filePathProcessedStocks = "ProcessedStocksHigh5Low2HM20", highestLimit =  0.05, lowestLimit = 0.02, hm_days = 20, name="ProcessedStocksHigh5Low2HM20")
strategyList.append(strategy)
strategy = PercentForPeriodStrategy.PercentForPeriodStrategy(isIndicatorLongList = False,
                                                             filePathProcessedStocks = "ProcessedStocksHigh10Low2HM30",
                                                             highestLimit =  0.1, lowestLimit = 0.02, hm_days = 30, name="ProcessedStocksHigh10Low2HM30")
strategyList.append(strategy)

#stocks = ['BRK.B','MMM' ]
#stocks = ['AMD','NVDA', 'AAPL','MMM','ACN','CAT','FIS' , 'NFLX']

reloadALL()
print("END")


