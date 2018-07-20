import abc

import pandas as pd
import numpy as np
import stockstats
import os

from .AbstractStrategy import AbstractStrategy

class AbstractSpikeForPeriodStrategy(AbstractStrategy):
    def Reload(self):
        self._featureList = ['adj close', 'Rev_1d', 'Rev_2d', 'Rev_3d', 'Rev_4d', 'Rev_5d', 'Rev_6d', 'Rev_7d']
    def process_data_for_labels(self, dfdata):
        hm_days = self.Hm_days
        df = dfdata.copy()

        for i in range(1,hm_days+1):
            df['_{}d'.format(i)] = (df['adj close'].shift(-i) -df['adj close'].shift(-i + 1)) / df['adj close'].shift(-i + 1)
        return df

    def buy_sell_hold(self, *args):
        cols = [c for c in args]
        # requirment = 0.02
        # if the spike is up and bigger than HighestLimit - sell
        # if the spike is down and bigger than LowestLimit - buy
        # sum = 0.0
        # if (cols[0][0] < -self.SpikeLowest):
        #     for col in cols[0][1:]:
        #         sum += col
        #         if sum > self.HighestLimit:
        #             return 1
        #         if col < -self.LowestLimit:
        #             return -1
        #     return 0

        if cols[0] < -self.SpikeLowest:
            s = 0.0
            for x in cols[1:]:
                s = s + x
                if (s > self.HighestLimit):
                    return 1
                if (s < -self.LowestLimit):
                    return -1
            return 0
            # s = sum(cols[1:])
            # if (s > self.HighestLimit):
            #     return 1
            # else:
            #     if (s < -self.LowestLimit):
            #         return -1
            #     else:
            #         return 0

        #if cols[0] > self.SpikeHighest:
         #   return 0
        return 0
