import abc
import pandas as pd
import numpy as np

from GemAIModule.DataFrameUtils import DataFrameUtils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)




class AbstractStrategy(object , metaclass=abc.ABCMeta):
    """
    Declare an interface for a type of product object.
    """

    def __init__(self,  name = "ProcessedStocks",isIndicatorLongList = False, filePathProcessedStocks = "ProcessedStocks", filePathIndexes = "Indexes",
                 highestLimit =  0.02, lowestLimit = 0.02, hm_days = 7, isDaily = True, spikeLowest = 0.05, spikeHighest = 0.05,
                 clf_name = "RF",
                  clf_test =  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                      max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
                      verbose=0, warm_start=False),
                 clf_actual=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                            max_depth=None, max_features='auto', max_leaf_nodes=None,
                                            min_samples_leaf=1,
                                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                                            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
                                            verbose=0, warm_start=False)
                 ):
        self._isDaily = isDaily
        self._name = name
        self._hm_days = hm_days
        self._highestLimit =  highestLimit
        self._lowestLimit = lowestLimit
        self._spikeLowest = spikeLowest
        self._spikeHighest = spikeHighest
        self._clf_test = clf_test
        self._clf_actual = clf_actual
        self._clf_name = clf_name
        if (isDaily == True):
            self._spFileLocation = 'sp500_joined_closes.csv'
        else:
            self._spFileLocation = 'sp500_weekly_joined_closes.csv'
        self._indexesList = ['^DJI', '^GDAXI', '^HSI', '^FCHI', '^GSPC', '^IXIC', '^N225', '^RUT', '^TYX']
        self._filePathIndexes = filePathIndexes
        if (isDaily == True):
            self._filePathProcessedStocks = filePathProcessedStocks
        else:
            self._filePathProcessedStocks = "Weekly_" + filePathProcessedStocks
        self._filePathStocks = "stock_dfs"
        if (isIndicatorLongList == False):
            self._indicators = ['atr', 'boll', 'boll_ub','boll_lb','open_2_sma','cr-ma3', 'volume_-3~1_min',
                                'vr_6_sma','cr-ma2','trix_9_sma','volume_-3,2,-1_max','vr',
                                'macds','adxr','cr-ma1','dma']
            #self._indicators = ['atr', 'boll']

        else:
            self._indicators = ['volume_delta', 'open_-2_r', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 'volume_-3,2,-1_max', 'volume_-3~1_min',
                      'kdjk', 'kdjd', 'kdjj', 'open_2_sma', 'macd', 'macds', 'macdh', 'boll', 'boll_ub', 'boll_lb',
                      'close_10.0_le_5_c', 'cr-ma2_xu_cr-ma1_20_c', 'rsi_6', 'rsi_12', 'wr_10', 'wr_6', 'cci', 'cci_20', 'tr',
                      'atr', 'dma', 'pdi', 'mdi', 'dx', 'adx', 'adxr', 'trix', 'trix_9_sma', 'vr', 'vr_6_sma']

        self._baseFileFolder = os.path.dirname(__file__)

        self._featureList = ['adj close', 'atr', 'boll_ub', 'cr-ma3', 'volume_-3~1_min',
                             'vr_6_sma', 'cr-ma2', 'trix_9_sma', 'volume_-3,2,-1_max', 'vr',
                             'macds', 'adxr', 'cr-ma1', 'dma']
        print("AbstractStrategy")

    @property
    def IsDaily(self):
        return self._isDaily

    @property
    def Hm_days(self):
        return self._hm_days

    @property
    def Name(self):
        return self._name

    @property
    def HighestLimit(self):
        return self._highestLimit

    @property
    def SpikeLowest(self):
        return self._spikeLowest

    @property
    def SpikeHighest(self):
        return self._spikeHighest

    @property
    def LowestLimit(self):
        return self._lowestLimit


    @property
    def IndexesList(self):
        return self._indexesList


    @property
    def SPFileLocation(self):
        abs_file_path = os.path.join(self._baseFileFolder, "../" + self._spFileLocation)
        return abs_file_path

    @property
    def Indicators(self):
        return self._indicators

    @property
    def FilePathProcessedStocks(self):
        abs_file_path = os.path.join(self._baseFileFolder, "../" + self._filePathProcessedStocks)
        return abs_file_path

    @property
    def FilePathStocks(self):
        abs_file_path = os.path.join(self._baseFileFolder, "../" + self._filePathStocks)
        return abs_file_path


    @property
    def Clf_Name(self):
        return self._clf_name

    @property
    def BaseFileFolder(self):
        return self._baseFileFolder

    @abc.abstractmethod
    def ExtractLabels(self, dfdata):
        raise NotImplementedError("Please Implement this method")

    ''''@abc.abstractmethod
    def ProcessData(self, ticker):
        raise NotImplementedError("Please Implement this method")'''

    @abc.abstractmethod
    def process_data_for_labels(self, dfdata):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def buy_sell_hold(self,*args):
        raise NotImplementedError("Please Implement this method")

    def AddIndex(self, ticker, df):
        df2 = pd.read_csv(self._filePathIndexes + '\{}.csv'.format(ticker))
        df2.set_index('Date', inplace=True)
        df2.rename(columns={'Adj Close': ticker}, inplace=True)
        df2.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        df = df.join(df2, how='inner')

        df[ticker] = df[ticker].pct_change()
        df = df.replace([np.inf, -np.inf], 0)
        df.fillna(0, inplace=True)
        return df

    def CleanDF(self, dfdata):
        dfdata.fillna(0, inplace=True)
        dfdata = dfdata.replace([np.inf, -np.inf], np.nan)
        dfdata.dropna(how='all', inplace=True)
        return dfdata

    def SetTestClf(self):
        self._

    def SetActualClf(self):
        raise NotImplementedError("Please Implement this method")


    def ProcessTicker(self, filename , ticker, skipPredict = False, ):
        print("Processing using {} clf {} high {} low {} hm {}".format(self._name, self.Clf_Name, self._highestLimit, self._lowestLimit, self._hm_days))
        dfUtils = DataFrameUtils()
        df = dfUtils.GetFeaturesFromCSV(filename + ticker + ".csv", self._featureList)
        if (df is None):
            return None, None, None
        rows, columns = df.shape
        numoffeature = columns - 1
        #print(rows, columns)



        df = df[(df.T != 0).any()]


        y, df , pred = self.ExtractLabels(df)

        # writer = pd.ExcelWriter('X.xlsx')
        # df.to_excel(writer, 'Sheet1')
        # writer.save()
        # df.to_csv("{}.csv".format(ticker))

        X = df.ix[1:,:-1].values    # independent variables

        pred = pred.ix[:, :-1].values  # independent variables
        #print("X : {}".format(X.shape))

        y = np.delete(y, (0), axis=0)
        #print("y : {}".format(y.shape))



        X = preprocessing.StandardScaler().fit_transform(X)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


        clf_test = self._clf_test
        # clf_test =  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #     max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
        #     min_samples_split=2, min_weight_fraction_leaf=0.0,
        #     n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
        #     verbose=0, warm_start=False)

        clf_test.fit(X_train, y_train)
        y_pred = clf_test.predict(X_test)
        acc = clf_test.score(X_test, y_test)
        print("accuracy : {}".format(acc))


        ###################################
        from sklearn.metrics import classification_report, confusion_matrix
        #print(classification_report(y_test,y_pred))
        confusionmatrix = confusion_matrix(y_test,y_pred, labels=[-1, 0, 1] )

        final = [0]
        if (skipPredict == True):
            #print(confusionmatrix)
            return acc, confusionmatrix, final


        # clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #     max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
        #     min_samples_split=2, min_weight_fraction_leaf=0.0,
        #     n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
        #     verbose=0, warm_start=False)
        clf = self._clf_actual

        m = RFECV(clf, scoring='accuracy')
        m.fit(X, y)

        final = m.predict(pred)

        print("Final:{}".format(final))
        return acc, confusionmatrix, final