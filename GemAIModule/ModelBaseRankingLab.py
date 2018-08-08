# Auther : Kfir Arbel
# date : 6.8.2018
# ModelBaseRankingLab Class
# use to manually test diffrent strategy and rate it

from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from GemAIModule.DataFrameUtils import DataFrameUtils
from sklearn import preprocessing
from StrategyModule.PercentForPeriodStrategy import PercentForPeriodStrategy
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE


class ModelBaseRankingLab():
    def __init__(self):
        self._featureList = ['adj close','atr', 'boll_ub','cr-ma3', 'volume_-3~1_min',
                                'vr_6_sma','cr-ma2','trix_9_sma','volume_-3,2,-1_max','vr',
                               'macds','adxr','cr-ma1','dma']
        #self._featureList = ['adj close','atr', 'boll', 'boll_ub']
    def GetValScore(self, filename ):
        #names = boston["feature_names"]
        dfUtils = DataFrameUtils()

        df = dfUtils.GetFeaturesFromCSV(filename, self._featureList)
        rows, columns = df.shape
        numoffeature = columns - 1
        print(rows, columns)

        strategy = PercentForPeriodStrategy()
        #utils = DataFrameUtils()
        #adjClose = utils.GetOnlyOneColumn(df, "adj close")

        df = df[(df.T != 0).any()]


        y, df , pred = strategy.ExtractLabels(df)
        #print("pred:{}".format(pred))

        X = df.ix[:,:-1].values    # independent variables
        pred = pred.ix[:, :-1].values  # independent variables
        #X = df.iloc[1:, 0:-1]

        # print(list(df.columns.values))
        #y = df['target'].values  # dependent variables


        #print("x:{}".format(X[0]))
        #print("shape x:{}".format(X.shape))
        #print("shape y:{}".format(y.shape))
        # Normalize

        X = preprocessing.StandardScaler().fit_transform(X)

        from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        scores = []

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFECV

        m = RFECV(RandomForestClassifier(), scoring='accuracy')
        m.fit(X, y)
        #print("Score RFECV:{}".format(m.score(X_test, y_test)))
        # generate submission datasets
        #print("Support:{}".format(m.support_))
        #print("ranking:{}".format(m.ranking_))
        #m.predict()

        # print summaries for the selection of attributes
        #print(df.columns.values)
        #print(rfe.support_)
        #print(rfe.ranking_)

        #print("x:{}".format(X_train.shape))
        #print("pred:{}".format(pred.shape))

        final = m.predict(pred)
        print("Final:{}".format(final))




        my_df = pd.DataFrame(X)
        my_df.to_csv('outX.csv', index=False, header=False)
        #my_df = pd.DataFrame(y)
        #my_df.to_csv('outY.csv', index=False, header=False)

        #for i in range(X.shape[1]):
         #   score = cross_val_score(rfe, X[:, i:i + 1], y, scoring="r2",
          #                          cv=ShuffleSplit(len(X), 3, .3))
           # scores.append((round(np.mean(score), 3), self._featureList[i]))



        #print(sorted(scores, reverse=True))

        #getstockcorr(df)




def getstockcorr(df):

        corr = df.corr()#   .to_csv("corr.csv")
        dftest = corr[(corr.abs()>0.8) & (corr.abs() < 1.0)]
        dftest.to_csv('corr2.csv')

        flat_cm = dftest.stack().reset_index()
        flat_cm['A_vs_B'] = flat_cm.level_0 + '_' + flat_cm.level_1
        flat_cm.columns = ['A', 'B', 'correlation', 'A_vs_B']
        flat_cm = flat_cm.loc[flat_cm.correlation < 1, ['A_vs_B', 'correlation']]
        print(flat_cm)


try:
    lab = ModelBaseRankingLab()
    lab.GetValScore("stock_ndx_processed\\processed_AMZN.csv")
    print("END")
except ValueError  as inst:
    print(type(inst))  # the exception instance
    print(inst.args)  # arguments stored in .args
    print(inst)  # __str__ allows args to be printed directly,
    print("args:", inst.args[0])