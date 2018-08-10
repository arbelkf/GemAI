# Auther : Kfir Arbel
# date : 6.8.2018
# Main Class

# Import Module - imports all or specific - stock or indexes
# StrategyModule - diffrent strategies to decide about the labels
# GemAIModule - runs diffrent kinds of prediction procedures
from GemAIModule.Predict import Predict
from GemAIModule.PredictCompose import PredictCompose
import sys

print("Write \"pred ticker\" - to predict specific ticker")
print("Write \"composed ticker\" - to predict specific ticker")
str = sys.stdin.readline()
while(str != "END\n"):
    strparam = str.split()
    if (strparam[0] == "pred"):
        pred = Predict()
        pred.PredictAll(skipPredict=False, firstStockIndex=7, lastStockIndex=7, specificTicker=strparam[1])
        print("END")
    if (strparam[0] == "composed"):
        pred = PredictCompose()
        pred.PredictAllForTicker(ticker=strparam[1])
        print("END")
    str = sys.stdin.readline()
