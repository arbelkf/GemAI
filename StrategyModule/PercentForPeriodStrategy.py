
import pandas as pd
import numpy as np
import stockstats
import os

from .AbstractStrategy import AbstractStrategy

from .AbstractPercentForPeriodStrategy import AbstractPercentForPeriodStrategy

class PercentForPeriodStrategy(AbstractPercentForPeriodStrategy):

    # def process_data_for_labels(self, dfdata):
    #     hm_days = self.Hm_days
    #
    #     df = dfdata.copy()
    #
    #     for i in range(1,hm_days+1):
    #         df['_{}d'.format(i)] = (df['adj close'].shift(-i) -df['adj close']) / df['adj close']
    #
    #     return df

    # def buy_sell_hold(self,*args):
    #     cols = [c for c in args]
    #     #requirment = 0.02
    #     for col in cols:
    #         if col > self.HighestLimit:
    #             return 1
    #         if col < -self.LowestLimit:
    #             return -1
    #     return 0

    def ExtractFeatures(self, dfdata):
        hm_days = self.Hm_days
        for i in range(1, hm_days + 1):
            dfdata['Rev_{}d'.format(i)] = (dfdata['adj close'].shift(i) - dfdata['adj close'].shift(i-1)) / dfdata['adj close'].shift(i-1)

        return dfdata


    def ExtractLabels(self, dfdata):
        df = self.process_data_for_labels(dfdata)
        #print(df.iloc[-1:])
        dfdata = self.ExtractFeatures(dfdata)
        hm_days = self.Hm_days
        #create future revenue
        #for i in range(1,hm_days+1):
        #    df['Rev_{}d'.format(i)] = (df['adj close'].shift(-i) -df['adj close']) / df['adj close']

        lst = []
        for x in range(1, self.Hm_days + 1):
            lst.append(df['_{}d'.format(x)])


        dfdata['target'] = list(map(self.buy_sell_hold, *lst))

        #df.dropna(inplace=True)
        dfdata = self.CleanDF(dfdata)

        dfdata['target'] = dfdata['target'].shift(-1)


        del dfdata['adj close']


        # save last row for prediction
        pred = dfdata.iloc[-1:]
        #print(pred)
        # drop the last row - cause the is no y there and it anyhow save for prediction
        dfdata = dfdata.drop(dfdata.index[len(dfdata) - 1])
        y = dfdata['target'].values
        return y, dfdata, pred


''' def ProcessData(self, ticker):
    dfdata = pd.read_csv(self.FilePathStocks + '\{}.csv'.format(ticker))
    #print(list(dfdata.columns.values))

    for index in self.IndexesList:
        df = self.AddIndex(index, df)

    y, df = self.ExtractLabels(ticker, dfdata)

    #print(list(df.columns.values))
    df['target'] = df['target'].shift(-1)

    # add indexes
    df2 = pd.DataFrame()
    df2 = pd.concat([df2, df[['Adj Close']]], axis=1)
    hm_days = self.Hm_days
    #for i in range(1, hm_days):
    #    df2[['_{}d'.format(i)]] = df[['PCT_Chg']].shift(-i)


    # add indicators from StockDataFrame

    stock = stockstats.StockDataFrame.retype(dfdata)
    for indicator in self.Indicators:
        df['{}'.format(indicator)] = stock['{}'.format(indicator)]
        df2 = pd.concat([df2, df[['{}'.format(indicator)]]], axis=1)

    df2 = pd.concat([df2, df[['target']]], axis=1)

    df2.dropna(inplace=True)
    df2.to_csv(self._filePathProcessedStocks + '\processed_{}.csv'.format(ticker))'''


