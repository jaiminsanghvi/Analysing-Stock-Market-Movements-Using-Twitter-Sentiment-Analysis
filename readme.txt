Instruction to run:

Befor running a given application user must install libraries and follow some instruction to collect live dataset
- pip install tweepy
- pip install yahoo_finance
- pip install oauth2
- pip install datetime

We are restricted to collect twitter historical dataset older than week. Hence it must need to update basic parameters in DownloadTwitterData.py file
- User must need to update current date in line [twitterData = get_twitter_data.TwitterData('2016-04-18')] to collect twitter corpus for a week.
- User must need to update according date range in line [yahooData = get_yahoo_data.YahooData('2016-04-10', "2016-04-17")] to collect yahoo_finance historical dataset

1. Load project in PyCharm IDE

2. Run DownloadTwitterData.py file: It contains application chain such as download twitter data, download yahoo data, create feature set, design feature matrix, analyze and predict tweet sentiment using Naïve Bayes classification and Support vector machine 
  
  Example:
  Run following command to download yahoo-stock and twitter data for specific company. 
  python DownloadTwitterData.py AAPL(Company stock keyword)
 

3. Run StockPrediction.py file: It contains basic code for stock prediction analysis. We have predicted stock market movement using Support vector machine model

4. The application run on live dataset hence it must need an internet connectivity.

5. In addition, it must need twitter authentication parameters to collect twitter data. Hence configuration file is required to run an application

