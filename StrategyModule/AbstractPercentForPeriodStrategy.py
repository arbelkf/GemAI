import abc

import pandas as pd
import numpy as np
import stockstats
import os

from .AbstractStrategy import AbstractStrategy

class AbstractPercentForPeriodStrategy(AbstractStrategy):
    def process_data_for_labels(self, dfdata):
        hm_days = self.Hm_days
        df = dfdata.copy()

        for i in range(1,hm_days+1):
            df['_{}d'.format(i)] = (df['adj close'].shift(-i) -df['adj close']) / df['adj close']
        return df

    def buy_sell_hold(self, *args):
        cols = [c for c in args]
        # requirment = 0.02
        for col in cols:
            if col > self.HighestLimit:
                return 1
            if col < -self.LowestLimit:
                return -1
        return 0
