
import pandas as pd
import numpy as np
import stockstats
import os

from .AbstractStrategy import AbstractStrategy
from .AbstractSpikeForPeriodStrategy import AbstractSpikeForPeriodStrategy

class SpikeForPeriodStrategy(AbstractSpikeForPeriodStrategy):

    def ExtractFeatures(self, dfdata):
        hm_days = self.Hm_days
        for i in range(1, hm_days + 1):
            dfdata['Rev_{}d'.format(i)] = (dfdata['adj close'].shift(i) - dfdata['adj close'].shift(i-1)) / dfdata['adj close'].shift(i-1)
        return dfdata


    def ExtractLabels(self, dfdata):

        dfdata = self.process_data_for_labels(dfdata)
        # print(df.iloc[-1:])

        for index in self.IndexesList:
            dfdata = self.AddIndex(index, dfdata)

        #dfdata['adj close'] = (dfdata['adj close'] / dfdata['^IXIC'])

        dfdata = self.ExtractFeatures(dfdata)


        hm_days = self.Hm_days

        lst = []
        for x in range(1, self.Hm_days + 1):
            lst.append(dfdata['_{}d'.format(x)])

        dfdata['target'] = list(map(self.buy_sell_hold, *lst))
        #dfdata['target'] = self.buy_sell_hold(*lst)

        #df.dropna(inplace=True)
        dfdata = self.CleanDF(dfdata)

        #dfdata.to_csv("final_{}".format(self.Name))
        dfdata['target'] = dfdata['target'].shift(-1)


        #dfdata['Max'] = dfdata[['_1d', '_2d', '_3d','_4d', '_5d', '_6d','_7d', '_8d', '_9d','_10d', '_11d', '_12d','_13d', '_14d']].max(axis=1)
        #dfdata.to_csv("teste.csv".format())

        writer = pd.ExcelWriter('output.xlsx')
        dfdata.to_excel(writer, 'Sheet1')
        writer.save()


        # delete Adj Close - I need only the change from yesterdays
        del dfdata['adj close']
        #del dfdata['Max']
        for x in range(1, self.Hm_days + 1):
            del dfdata['_{}d'.format(x)]

        # save last row for prediction
        pred = dfdata.iloc[-1:]
        #print(pred)
        # drop the last row - cause the is no y there and it anyhow save for prediction

        dfdata = dfdata.drop(dfdata.index[len(dfdata) - 1])
        dfdata = self.CleanDF(dfdata)

        y = dfdata['target'].values

        return y, dfdata, pred

