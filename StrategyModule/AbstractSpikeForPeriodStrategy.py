# Auther : Kfir Arbel
# Abstract Spike For Period Stratgegy -

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

        # value of each "_{i}d" will be the change from the day before it
        for i in range(1,hm_days+1):
            df['_{}d'.format(i)] = (df['adj close'].shift(-i) -df['adj close'].shift(-i + 1)) / df['adj close'].shift(-i + 1)
        return df

    # calculates the labels from the hm days before
    def buy_sell_hold(self, *args):
        cols = [c for c in args]

        # once the spike is lower than the SpikeLowest value :
        # if the next 1:hm change of values will go above HighestLimit - return 1
        # if the next 1:hm change of values will go under LowestLimit - return -1
        # if the next 1:hm change of values will be in between values - return 0
        if cols[0] < -self.SpikeLowest:
            s = 0.0
            for x in cols[1:]:
                s = s + x
                if (s > self.HighestLimit):
                    return 1
                if (s < -self.LowestLimit):
                    return -1
            return 0

        return 0
