import pandas as pd
import os

class DataFrameUtils():
    def GetFeaturesFromCSV(self, fileName, featureList):
        try:
            if (os.path.isfile(fileName)):
                df = pd.read_csv(fileName)
                df2 = pd.DataFrame()
                df.set_index('Date', inplace=True)
                #df2 = pd.concat([df2, df[['pct_1d']]], axis=1)
                for index in featureList:
                    df2 = pd.concat([df2, df[[index]]], axis=1)

                #df2 = pd.concat([df2, df[['adj close']]], axis=1)

                my_df = pd.DataFrame(df2)
                #my_df.to_csv('dfInternal.csv', index=False, header=False)

                df2.dropna(inplace=True)
                #print(df2.head())
                return df2
            else:
                print("file {} missing:".format(fileName))
                return None
        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])

    def GetOnlyOneColumn(self, df, columnName):
        try:
            df2 = pd.DataFrame()
            #df.set_index('Date', inplace=True)
            df2 = pd.concat([df2, df[[columnName]]], axis=1)

            df2.dropna(inplace=True)
            return df2

        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])
