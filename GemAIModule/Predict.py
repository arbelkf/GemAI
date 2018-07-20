from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from GemAIModule.DataFrameUtils import DataFrameUtils
from sklearn import preprocessing
from StrategyModule.PercentForPeriodStrategy import PercentForPeriodStrategy
from StrategyModule.PercentForPeriodIndexCorrStrategy import PercentForPeriodIndexCorrStrategy
from StrategyModule.PercentForPeriodIndexLogCorrStrategy import PercentForPeriodIndexLogCorrStrategy
from StrategyModule.PercentForPeriodIndexCorrRollingStrategy import PercentForPeriodIndexCorrRollingStrategy
from StrategyModule.Context import Context
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)



class Predict(object):

    def __init__(self):
        print("Start...")
        self._df = pd.DataFrame()

        list_of_params = [(0.2, 0.2, 14), (0.2, 0.15, 14), (0.1, 0.1, 14), (0.2, 0.1, 14), (0.1, 0.05, 14)]
        #for high in range(1, 20, 1):
         #   for low in range (1, 20, 1):
          #      for days in range (4, 60, 10):
           #         list_of_params.append((high * 0.01, low * 0.01, days))

        clf_test_RF =  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

        clf_actual_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

        clf_test_KNN = KNeighborsClassifier(n_neighbors=3)
        clf_actual_KNN = KNeighborsClassifier(n_neighbors=3)
        clf_test_Svc = SVC()
        clf_actual_Svc = SVC()
        clf_test_Gradient = GradientBoostingClassifier(n_estimators=100)
        clf_actual_Gradient = GradientBoostingClassifier(n_estimators=100)

        self._StrategyList = []
        for item in list_of_params:
            str = PercentForPeriodIndexCorrStrategy(name="PercentForPeriodIndexCorrStrategy",
                                                    highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                    filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_RF, clf_actual = clf_actual_RF, clf_name="RF")
            self._StrategyList.append(str)
            str = PercentForPeriodStrategy(name="PercentForPeriodStrategy",
                                                    highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                    filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_RF, clf_actual = clf_actual_RF, clf_name="RF")
            self._StrategyList.append(str)
            str = PercentForPeriodIndexCorrRollingStrategy(name="PercentForPeriodIndexCorrRollingStrategy",
                                           highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                           filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_RF, clf_actual = clf_actual_RF, clf_name="RF")
            self._StrategyList.append(str)
            str = PercentForPeriodIndexLogCorrStrategy(name="PercentForPeriodIndexLogCorrStrategy",
                                                           highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                           filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_RF, clf_actual = clf_actual_RF, clf_name="RF")
            self._StrategyList.append(str)
            # str = PercentForPeriodIndexCorrStrategy(name="PercentForPeriodIndexCorrStrategy",
            #                                         highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                         filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_KNN, clf_actual = clf_actual_KNN, clf_name="KNN")
            # self._StrategyList.append(str)
            # str = PercentForPeriodStrategy(name="PercentForPeriodStrategy",
            #                                         highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                         filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_KNN, clf_actual = clf_actual_KNN, clf_name="KNN")
            # self._StrategyList.append(str)
            # str = PercentForPeriodIndexCorrRollingStrategy(name="PercentForPeriodIndexCorrRollingStrategy",
            #                                highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_KNN, clf_actual = clf_actual_KNN, clf_name="KNN")
            # self._StrategyList.append(str)
            # str = PercentForPeriodIndexLogCorrStrategy(name="PercentForPeriodIndexLogCorrStrategy",
            #                                                highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                                filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_KNN, clf_actual = clf_actual_KNN, clf_name="KNN")
            # self._StrategyList.append(str)
            #
            # str = PercentForPeriodIndexCorrStrategy(name="PercentForPeriodIndexCorrStrategy",
            #                                         highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                         filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Svc,
            #                                         clf_actual=clf_actual_Svc, clf_name="SVC")
            # self._StrategyList.append(str)
            # str = PercentForPeriodStrategy(name="PercentForPeriodStrategy",
            #                                highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Svc,
            #                                         clf_actual=clf_actual_Svc, clf_name="SVC")
            # self._StrategyList.append(str)
            # str = PercentForPeriodIndexCorrRollingStrategy(name="PercentForPeriodIndexCorrRollingStrategy",
            #                                                highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                                filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Svc,
            #                                         clf_actual=clf_actual_Svc, clf_name="SVC")
            # self._StrategyList.append(str)
            # str = PercentForPeriodIndexLogCorrStrategy(name="PercentForPeriodIndexLogCorrStrategy",
            #                                            highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                            filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Svc,
            #                                         clf_actual=clf_actual_Svc, clf_name="SVC")
            # self._StrategyList.append(str)

            str = PercentForPeriodIndexCorrStrategy(name="PercentForPeriodIndexCorrStrategy",
                                                    highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                    filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Gradient,
                                                    clf_actual=clf_actual_Gradient, clf_name="Gradient")
            self._StrategyList.append(str)
            str = PercentForPeriodStrategy(name="PercentForPeriodStrategy",
                                           highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                           filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Gradient,
                                                    clf_actual=clf_actual_Gradient, clf_name="Gradient")
            self._StrategyList.append(str)
            str = PercentForPeriodIndexCorrRollingStrategy(name="PercentForPeriodIndexCorrRollingStrategy",
                                                           highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                           filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Gradient,
                                                    clf_actual=clf_actual_Gradient, clf_name="Gradient")
            self._StrategyList.append(str)
            str = PercentForPeriodIndexLogCorrStrategy(name="PercentForPeriodIndexLogCorrStrategy",
                                                       highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                       filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Gradient,
                                                    clf_actual=clf_actual_Gradient, clf_name="Gradient")
            self._StrategyList.append(str)

    def DoStrategy(self, strategy, ticker, skipPredict = True):
        context = Context(strategy)
        acc, confusionmatrix, final = context.ProcessTicker("stock_ndx_processed\\processed_", ticker, skipPredict)
        return acc, confusionmatrix, final

    def PredictTicker(self, ticker):

        try:
            #pred = Predict()
            print("Processing:{}".format(ticker))

            #self.DoStrategy(self._strategy, ticker)

            #self.DoStrategy(self._strategyCorr, ticker)

            #self.DoStrategy(self._strategyCorrRolling, ticker)

            #self.DoStrategy(self._strategyLogCorr, ticker)

            #self.DoStrategy(self._strategyCorrDown5Up15, ticker)

            #self.DoStrategy(self._strategyCorrDown10Up25, ticker)

            labels = ['ticker' ,'name', 'clf','acc', 'BPSemi','BPAqurate', 'final','highest', 'lowest', 'Hm_days', '_-1 ', '_0 ', '_1 ', '_(0,1)',
                      '_(0,2) ', '_(1,0)', '_(1,2)', '_(2,0)', '_(2,1)']

            for item in self._StrategyList:
                acc, confusionmatrix, final = self.DoStrategy(item, ticker, skipPredict=False)
                #newLine = np.array([[confusionmatrix[1,1]]])
                if (acc == None):
                    return None
                x,y = confusionmatrix.shape
                BuyPercentAcc = 0
                if (confusionmatrix[0, 2] +  confusionmatrix[2, 0] + confusionmatrix[2, 2] != 0):
                    BuyPercentAcc = confusionmatrix[2, 2] / (confusionmatrix[0, 2] +  confusionmatrix[2, 0] + confusionmatrix[2, 2])

                BuyPercentAccAqurate = confusionmatrix[2, 2] / (
                        confusionmatrix[0, 2] + confusionmatrix[2, 0] +
                        confusionmatrix[0, 1] + confusionmatrix[1, 0] +
                        confusionmatrix[2, 2])
                if (x > 2):
                    c = np.array(
                        [[ticker, item.Name, item.Clf_Name, acc, BuyPercentAcc,BuyPercentAccAqurate, final[0], item.HighestLimit, item.LowestLimit, item.Hm_days, confusionmatrix[0, 0],
                          confusionmatrix[1, 1], confusionmatrix[2, 2], confusionmatrix[0, 1], confusionmatrix[0, 2],
                          confusionmatrix[1, 0],
                          confusionmatrix[1, 2], confusionmatrix[2,0], confusionmatrix[2, 1]]])
                    if (len(self._df) > 0 ):
                        self._df = self._df.append(pd.DataFrame(c, columns=labels))
                    else:
                        self._df = pd.DataFrame(c,  columns=labels)
                else:
                    print("Skipped...")

            # #filename = "strategyparams_processed\\complete.xlsx"
            #
            # writer = pd.ExcelWriter(filename)
            # df.to_excel(writer, '{}'.format(ticker))
            # writer.save()

        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])

    def PredictAll(self, filepath = '..\ImportModule\\ndx.csv'):
        data = pd.read_csv(filepath)
        tickers = data['ticker']


        for ticker in tickers[:]:
            sys.stdout.write('.')
            self.PredictTicker(ticker)

        filename = "strategyparams_processed\\complete.xlsx"

        writer = pd.ExcelWriter(filename)
        self._df.to_excel(writer, '{}'.format(ticker))
        writer.save()

pred = Predict()
pred.PredictAll()
print("END")


