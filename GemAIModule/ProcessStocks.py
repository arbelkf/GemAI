# Auther : Kfir Arbel
# date : 6.8.2018
# ProcessStocks Class

import pandas as pd
import numpy as np
import stockstats
import os
import sys
import pickle

from StrategyModule.PercentForPeriodStrategy import PercentForPeriodStrategy

class ProcessStocks:

    _indexesList = ['^DJI', '^GDAXI', '^HSI', '^FCHI', '^GSPC', '^IXIC', '^N225', '^RUT', '^TYX']
    _indicators = ['volume_delta', 'open_-2_r', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 'volume_-3,2,-1_max',
                        'volume_-3~1_min',
                        'kdjk', 'kdjd', 'kdjj', 'open_2_sma', 'macd', 'macds', 'macdh', 'boll', 'boll_ub', 'boll_lb',
                        'close_10.0_le_5_c', 'cr-ma2_xu_cr-ma1_20_c', 'rsi_6', 'rsi_12', 'wr_10', 'wr_6', 'cci',
                        'cci_20', 'tr',
                        'atr', 'dma', 'pdi', 'mdi', 'dx', 'adx', 'adxr', 'trix', 'trix_9_sma', 'vr', 'vr_6_sma']

    def ProcessData(self, ticker, daily = True):
        strategy = PercentForPeriodStrategy()
        strategy.process_data_for_labels()

    def ProcessAll(self, filePathOriginal, filePathDest , daily = True):
        if not os.path.exists(filePathDest):
            os.makedirs(filePathDest)
        data = pd.read_csv("..\\ImportModule\\ndx.csv")
        tickers = data['ticker']
        for ticker in tickers[:]:
            sys.stdout.write('.')
            if (daily == True):
                filename = '\\{}.csv'.format(ticker)
            else:
                filename = '\\_weekly\\{}.csv'.format(ticker)
            self.ProcessStock(filename, ticker,filePathOriginal, filePathDest )

    def ProcessAllDfs(self, filePathOriginal, filePathDest, daily=True):
        if not os.path.exists(filePathDest):
            os.makedirs(filePathDest)
        with open("..\\ImportModule\\sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
        for ticker in tickers[:]:
            sys.stdout.write('.')
            if (daily == True):
                filename = '\\{}.csv'.format(ticker)
            else:
                filename = '\\_weekly\\{}.csv'.format(ticker)
            self.ProcessStock(filename, ticker, filePathOriginal, filePathDest)

    def ProcessStock(self, filename, ticker, filePathOriginal, filePathDest):
        try:
            filename = "..\\ImportModule\\" + filePathOriginal + filename
            if (os.path.exists(filename)):
                print("Proccessing {}".format(ticker))
            else:
                print("Skipping {}".format(ticker))
                return
            df = pd.read_csv(filename)
            df.set_index('Date', inplace=True)
            hm_days = 15
            '''for i in range(1,hm_days+1):
                df['PCT_{}d'.format(i)] = (df['Adj Close'] -df['Adj Close'].shift(i)) / (df['Adj Close']).shift( i)
                df.fillna(0,inplace=True)'''

            stock = stockstats.StockDataFrame.retype(df)
            for indicator in self._indicators:
                df['{}'.format(indicator)] = stock['{}'.format(indicator)]
                #df2 = pd.concat([df2, df[['{}'.format(indicator)]]], axis=1)


            for index in self._indexesList:
                df = self.AddIndex("Indexes", index, df)


            df.dropna(inplace=True)
            df.to_csv(filePathDest + '\processed_{}.csv'.format(ticker))
        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])

    def AddIndex(self, filePathIndexes, ticker, df):
        df2 = pd.read_csv("..\\ImportModule\\" + filePathIndexes + '\{}.csv'.format(ticker))
        df2.set_index('Date', inplace=True)
        df2.rename(columns={'Adj Close': ticker}, inplace=True)
        df2.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        df = df.join(df2, how='outer')

        df[ticker] = df[ticker].pct_change()
        df = df.replace([np.inf, -np.inf], 0)
        df.fillna(0, inplace=True)
        return df

process = ProcessStocks()
process.ProcessAll('stock_ndx', 'stock_ndx_processed' )
process.ProcessAll('stock_dfs', 'stock_dfs_processed' )
print("END")