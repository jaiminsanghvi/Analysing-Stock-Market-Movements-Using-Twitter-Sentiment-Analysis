from yahoo_finance import Share



class YahooData:
    def __init__(self, startDate, endDate):
        self.startDate = startDate
        self.endDate = endDate

    def getYahooData(self, keyword):
        yahoo = Share(keyword)
        historical_data = yahoo.get_historical(self.startDate, self.endDate)
        return historical_data

#historical_data =
#print historical_data

#for i in range(len(historical_data)):
#    print historical_data[i]
#    print historical_data[i]['Symbol'], " Open : ",historical_data[i]['Open'], " Close : ",historical_data[i]['Close']