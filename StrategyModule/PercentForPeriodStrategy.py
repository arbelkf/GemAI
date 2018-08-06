
import pandas as pd
import numpy as np
import stockstats
import os

from .AbstractStrategy import AbstractStrategy

from .AbstractPercentForPeriodStrategy import AbstractPercentForPeriodStrategy

class PercentForPeriodStrategy(AbstractPercentForPeriodStrategy):


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


