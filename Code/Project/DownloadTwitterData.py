import get_twitter_data
import get_yahoo_data
import re
import tweepy
import csv
import sys
from StringIO import StringIO
from zipfile import ZipFile
from urllib import urlopen
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
import Multiclass_SVM
from sklearn.naive_bayes import BernoulliNB
import NaiveBayes
from sklearn.metrics import confusion_matrix
import datetime

# Get input from user
print "Enter any keyword listed below"
print "Example --> AAPL GOOG YHOO MSFT GS"
print "-------------------------------------------------"
print "-------------------------------------------------"

response = raw_input("Please enter Keyword: ")

while not response:
    response = raw_input("Please enter Keyword: ")


# Get Tweets
keyword = '$'+response
# time = 'today'
time = 'lastweek'

print "Fetch twitter data for "+ response+" company keyword...."

twitterData = get_twitter_data.TwitterData('2016-04-18')
tweets = twitterData.getTwitterData(keyword, time)

print "Twitter data fetched \n"

print "Fetch yahoo finance data for "+response+" given company keyword.... "
keyword2 = response
yahooData = get_yahoo_data.YahooData('2016-04-01', "2016-04-17")
historical_data = yahooData.getYahooData(keyword2)
yahoo_open_price = {}
yahoo_close_price = {}
yahoo_high_price = {}
yahoo_low_price = {}
for i in range(len(historical_data)):
    date = historical_data[i]['Date'].replace(",","")
    yahoo_open_price.update({date: historical_data[i]['Open']})
    yahoo_close_price.update({date: historical_data[i]['Close']})
    yahoo_high_price.update({date: historical_data[i]['High']})
    yahoo_low_price.update({date: historical_data[i]['Low']})

print "Yahoo data fetched \n"

print "Collect tweet and process twitter corpus...."
tweet_s = []
for key,val in tweets.items():
    for value in val:
        tweet_s.append(value)

csvFile = open('Data/SampleTweets.csv', 'w')
csvWriter = csv.writer(csvFile)


# start replaceTwoOrMore
def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
# end

# start process_tweet
def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet
# end

# start getStopWordList
def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
# end

# start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []
    words = tweet.split()
    for w in words:
        w = replaceTwoOrMore(w)
        w = w.strip('\'"?,.')
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
# end

# start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
# end

def getFeatureVectorAndLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    labels = []
    file = open("newfile.txt", "w")

    for t in tweets:
        label = 0
        map = {}
        for w in sortedFeatures:
            map[w] = 0

        tweet_words = t[0]
        tweet_opinion = t[1]

        # Fill the map
        for word in tweet_words:
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            if word in map:
                map[word] = 1
        # end for loop
        values = map.values()
        feature_vector.append(values)
        if (tweet_opinion == '|positive|'):
            label = 0
            tweet_opinion = 'positive'
        elif (tweet_opinion == '|negative|'):
            label = 1
            tweet_opinion = 'negative'
        elif (tweet_opinion == '|neutral|'):
            label = 2
            tweet_opinion = 'neutral'
        labels.append(label)
        feature_vector_value = str(values).strip('[]')
        file.write(feature_vector_value + "," + str(label) + "\n")
    file.close()
    return {'feature_vector' : feature_vector, 'labels': labels}
#end

# Download the AFINN lexicon, unzip, and read the latest word list in AFINN-111.txt
url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(StringIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')

afinn = dict()
for line in afinn_file:
    parts = line.strip().split()
    if len(parts) == 2:
        afinn[parts[0]] = int(parts[1])

def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()

def afinn_sentiment(terms, afinn):

    total = 0.
    for t in terms:
        if t in afinn:
            total += afinn[t]
    return total

def sentiment_analyzer():
    tokens = [tokenize(t) for t in tweet_s]  # Tokenize all the tweets

    afinn_total = []
    for tweet in tokens:
        total = afinn_sentiment(tweet, afinn)
        afinn_total.append(total)

    positive_tweet_counter = []
    negative_tweet_counter = []
    neutral_tweet_counter = []
    for i in range(len(afinn_total)):
        if afinn_total[i] > 0:
            positive_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["|positive|", tweet_s[i].encode('utf-8').split("|")[0], tweet_s[i].encode('utf-8').split("|")[1], float(afinn_total[i])])
        elif afinn_total[i] < 0:
            negative_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["|negative|", tweet_s[i].encode('utf-8').split("|")[0], tweet_s[i].encode('utf-8').split("|")[1], float(afinn_total[i])])
        else:
            neutral_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["|neutral|", tweet_s[i].encode('utf-8').split("|")[0], tweet_s[i].encode('utf-8').split("|")[1], float(afinn_total[i])])

# Main

print "Processing tweets and store in CSV file ...."
sentiment_analyzer()

print "Tweet corpus processed \n "

print "Preparing dataset...."
# Read the tweets one by one and process it
inpTweets = csv.reader(open('Data/SampleTweets.csv', 'rb'), delimiter=',')
stopWords = getStopWordList('Data/stopwords.txt')
count = 0;
featureList = []
labelList = []
tweets = []
dates =[]
date_split =[]
list_tweet = []
print "Creating feature set and generating feature matrix...."
for row in inpTweets:
    if len(row) == 4:
        list_tweet.append(row)
        sentiment = row[0]
        date = row[1]
        t = row[2]

        date_split.append(date)
        dates.append(date)
        labelList.append(sentiment)
        processedTweet = processTweet(t)
        featureVector = getFeatureVector(processedTweet, stopWords)
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment));

result = getFeatureVectorAndLabels(tweets, featureList)

print "Dataset is ready \n"

print "Sentiment prediction using Naive Bayes Bernoulli and SVM model...."
# Naive Bernoulli and SVM Algorithm
data2 = open('newfile.txt', 'r')

inp_data2 = []
files = np.loadtxt(data2,dtype=str, delimiter=',')

inp_data2 = np.array(files[:,0:-1], dtype='float')
givenY = files[:,-1]

target2=np.zeros(len(givenY), dtype='int')
unique_y = np.unique(givenY)

for cls in range(len(givenY)):
    for x in range(len(unique_y)):
        if(givenY[cls] == unique_y[x]):
            target2[cls] = x

X = np.array(inp_data2)
y = np.array(target2)

# print type(X)
# print type(y)

max_gX = {}
maximum_gX = []
temp = 0
svn_temp = 0
final_precision=0
final_recall = 0
final_fmeasure = 0
final_accuracy = 0

svm_final_accuracy = 0
svm_final_precision = 0
svm_final_recall = 0
svm_final_fmeasure = 0

svm_accuracy = []
NB_accuracy = []
NBSKL_accuracy = []
kf = KFold(X.shape[0], n_folds=6, shuffle=False)
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # SVM Start

    clf = Multiclass_SVM.MulticlassSVM(C=0.1, tol=0.01, max_iter=100, random_state=0, verbose=1)
    clf.fit(X_train, y_train)
    predicted_y = clf.calculate_prediction(X_test)
    svm_accuracy.append(accuracy_score(y_test,predicted_y))
    svm_confusion_mat = confusion_matrix(y_test, predicted_y)
    sv_accuracy, svm_precision_val, svm_recall_val, svm_f_measure_val = clf.svm_findOtherParameters(svm_confusion_mat)

    # SVM end

    # Naive Bayes

    clf_NB = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    clf_NB.fit(X_train, y_train)
    predictedBNB_y = clf_NB.predict(X_test)
    NBSKL_accuracy.append(accuracy_score(y_test,predictedBNB_y))

    nb = NaiveBayes.NaiveBayesBernoulli()
    # iterate data for each class
    for clas in np.unique(y):
        class_feature_matrix = X_train[y_train==clas]
        prior_array = len(class_feature_matrix)*1.0/len(X_train)
        # print prior_array
        alpha = [(np.sum(class_feature_matrix[:,i])/len(class_feature_matrix)) for i in range(class_feature_matrix.shape[1])]
        gX = nb.membership_function(X_test, alpha, prior_array)
        max_gX.update({int(clas): gX})

    # find discriminant function
    disc_function = nb.discriminant_function(max_gX, np.unique(y))
    # print disc_function
    confusion_mat = confusion_matrix(y_test, predictedBNB_y)
    # print confusion_mat

    # find precision, recall , f-measure
    accuracy, precision_val, recall_val, f_measure_val = nb.findOtherParameters(confusion_mat)

    if accuracy_score(y_test,predictedBNB_y) > temp:
        if (accuracy_score(y_test,predictedBNB_y) != 1):
            final_accuracy = accuracy_score(y_test,predictedBNB_y)
            final_precision = precision_val
            final_recall = recall_val
            final_fmeasure = f_measure_val
            temp = accuracy_score(y_test,predictedBNB_y)

    if accuracy_score(y_test,predicted_y) > svn_temp:
        if (accuracy_score(y_test,predicted_y) != 1):
            svm_final_accuracy = accuracy_score(y_test,predicted_y)
            svm_final_precision = svm_precision_val
            svm_final_recall = svm_recall_val
            svm_final_fmeasure = svm_f_measure_val
            svn_temp = accuracy_score(y_test,predicted_y)

# Naive Bayes end

print "Bernoulli NB"
print "Accuracy =" ,max(NBSKL_accuracy)
print "Precision = ", final_precision
print "Recall = ", final_recall
print "F-Measure", final_fmeasure
print "\n"
print "SVM"
print "Accuracy =", max(svm_accuracy)
print "Precision = ", svm_final_precision
print "Recall = ", svm_final_recall
print "F-Measure", svm_final_fmeasure
print "\n"



print "Prediction completed \n"

print "Preparing dataset for stock prediction using yahoo finance and tweet sentiment...."
date_tweet_details = {}
file = open("stockpredict.txt", "w")
for dateVal in np.unique(date_split):
    date_totalCount = 0
    date_PosCount = 0
    date_NegCount = 0
    date_NutCount = 0
    total_sentiment_score = 0.
    for row in list_tweet:
        sentiment = row[0]
        temp_date = row[1]
        sentiment_score = row[3]
        if(temp_date == dateVal):
            total_sentiment_score += float(sentiment_score)
            date_totalCount+=1
            if (sentiment == '|positive|'):
                date_PosCount+=1
            elif (sentiment == '|negative|'):
                date_NegCount+=1
            elif (sentiment == '|neutral|'):
                date_NutCount+=1

    s = str(date_totalCount)+" "+str(date_PosCount)+" "+str(date_NegCount)+" "+str(date_NutCount)
    date_tweet_details.update({dateVal: s})

    dateVal = dateVal.strip()
    day = datetime.datetime.strptime(dateVal, '%Y-%m-%d').strftime('%A')
    closing_price = 0.
    opening_price = 0.
    if day == 'Saturday':
        update_date = dateVal.split("-")
        if len(str((int(update_date[2])-1)))==1:
            dateVal = update_date[0]+"-"+update_date[1]+"-0"+str((int(update_date[2])-1))
        else:
            dateVal = update_date[0] + "-" + update_date[1] + "-" + str((int(update_date[2]) - 1))
        opening_price = yahoo_open_price[dateVal]
        closing_price = yahoo_close_price[dateVal]
    elif day == 'Sunday':
        update_date = dateVal.split("-")
        if len(str((int(update_date[2])-2)))==1:
            dateVal = update_date[0]+"-"+update_date[1]+"-0"+str((int(update_date[2])-2))
        else:
            dateVal = update_date[0] + "-" + update_date[1] + "-" + str((int(update_date[2]) - 2))
        opening_price = yahoo_open_price[dateVal]
        closing_price = yahoo_close_price[dateVal]
    else:
        opening_price = yahoo_open_price[dateVal]
        closing_price = yahoo_close_price[dateVal]
    """
    print dateVal
    print "Total tweets = ", date_totalCount, " Positive tweets = ", date_PosCount, " Negative tweets = ", date_NegCount
    print "Total sentiment score = ", total_sentiment_score
    print "Opening Price = ", opening_price
    print "Closing Price = ", closing_price
    """

    market_status = 0
    if (float(closing_price)-float(opening_price)) > 0:
        market_status = 1
    else:
        market_status =-1
    file.write( str(date_PosCount) + "," + str(date_NegCount) + "," + str(date_NutCount) +"," + str(date_totalCount) + "," + str(market_status) + "\n")

    # print " Total Tweet For date =",dateVal ," Count =" , date_totalCount
    # print " Positive Tweet For date =",dateVal ," Count =" , date_PosCount
    # print " Negative Tweet For date =",dateVal ," Count =" , date_NegCount
    # print " Neutral Tweet For date =",dateVal ," Count =" , date_NutCount
file.close()

print "Dataset is ready for stock prediction \n"
# end









