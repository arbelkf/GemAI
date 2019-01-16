# GemAI
Analysis and forecasting stock prices using different machine learning strategies and parameters:
In this project I have collected closing prices from all the main stocks in the USA for the last 25 years.
To each stock values I have added related markets indexes and indicators. Than I have run pattern recognitions on this data.

I have checked the following predictors:
* RandomForestClassifier
*KNeighborsClassifier 
* GradientBoostingClassifier
* SVC – support vector machine
* GradientBoostingClassifier
* Best predictor was the random forest predictor with the 14 days predictor and sliding window of 3 days.

It predicts buying opportunity.

The results are around 33 percent for true positive and true negative. The label is 3 options label (buy/sell/hold) – which means that randomly deciding will result also in 33 percent.

**Main.py** – main console to run

The Import Modlue is in charge of importing the raw data like daily stocks values and daily indexes values
**Scrapper** – scrap stocks data
**ScrapperIndex** – scrap indexes values

**Strategy Module** is in charge of implementing different strategies to calculate the labels from the raw data

**AbstractStrategy** – abstract class – implements:
* **ProcessSpecificTicker** – extract label, creates confusionmatrix and prediction for the next day
* **ProcessComposedTicker** – collects raw data from different stocks to one data table and run ProcessSpecificTickeron it. Under the assumption that all stocks behave the same, I create more raw data.
**AbstractPercentForPeriodStrategy** – abstract class – inherit from AbstractStrategy, implements strategy for change in percentage of the value of the stock for specified period
**PercentForPeriodIndexCorrRollingStrategy** – concrete class – inherit from AbstractPercentForPeriodStrategy, implements a rolling window to average an already divided by the index values
**PercentForPeriodIndexCorrStrategy** - concrete class – inherit from AbstractPercentForPeriodStrategy, values are divided by the index values
**PercentForPeriodIndexLogCorrStrategy** – concrete class – inherit from AbstractPercentForPeriodStrategy, implements a log on the already divided by the index values

**AbstractSpikeForPeriodStrategy** – abstract class – inherit from AbstractStrategy, implements strategy for sharp change in percentage of the value of the stock for specified period, once the level is crossed – the label is set
**SpikeForPeriodStrategy** – concrete class – inherit from AbstractSpikeForPeriodStrategy – implements looking for a rise in the percentage for a specific period 

**GemAImodule** – machine learning module:
**FeaturesLab** – a lab, searches for the covariance values between the different features 
**DataFrameUtils** – utils for handling the raw data
**ModelBaseRankingLab** – utils for searching for the best predictor
**Predict** -  the predictor
* **PredictTicker** – predict specific stock and returns also confusion matraix
* **PredictAll** – runs predict on all the stocks from Nasdaq(ndx)
**PredictSpikes** – predictor that uses the spike strategy
**PredictCompose** – predictor that runs throw all stocks in the Nasdaq index
