# Auther : Kfir Arbel
# Abstract Percent For Period Stratgegy
import abc

import pandas as pd
import numpy as np
import stockstats
import os

from .AbstractStrategy import AbstractStrategy

class AbstractPercentForPeriodStrategy(AbstractStrategy):
    # calculate the hm values
    def process_data_for_labels(self, dfdata):
        hm_days = self.Hm_days
        df = dfdata.copy()
        # value of each "_{i}d" will be the change of the current day to the base day value
        for i in range(1,hm_days+1):
            df['_{}d'.format(i)] = (df['adj close'].shift(-i) -df['adj close']) / df['adj close']
        return df

    # calculate the labels from the "_{i}d"
    # if the next hm change of values will go above HighestLimit - return 1
    # if the next hm change of values will go under LowestLimit - return -1
    # if the next hm change of values will be in between values - return 0
    def buy_sell_hold(self, *args):
        cols = [c for c in args]
        # requirment , default = 0.02
        for col in cols:
            if col > self.HighestLimit:
                return 1
            if col < -self.LowestLimit:
                return -1
        return 0
