# stockPrediction_CNN
Uisng CNN to predicte stock market trend, and feeding with 2D images

Stock market is a promising ﬁnancial investment that can generate great wealth. However, the volatile nature of the stock market makes it a very high risk investment. Thus, a lot of researchers have contributed their efforts to forecast the stock market pricing and average movement. Researchers have used various methods in computer science and economics in their quests to gain a piece of this volatile information and make great fortune out of the stock market investment. The aim is to predict the stock trend in the future. Therefore, this work introduced a CNN model fed with 2D images to predict stock trend 10 days late (future stock market forecasting). The best result of thismodelis58%whichisbetterthanPoulos’s(Poulos,2014)3-layerLSTM(55.45%) and SVM (56.31%) model.

# Data Collection 
In this experiment, images represented history data would be used as input. First, history data need to be collected. 3000 stock indexes are randomly chosen from Yahoo Finance. The history data of them in 15 years before 2017 were downloaded. Unfortunately,becauseofdatalossornetworkissues,only2397indexesweredownloaded. Then during pre-processing, about 400 indexes were removed because of some‘NAN’value. Beforeremovingthem,theclosedvaluehadbeenusedtorepresent these ‘NAN’ value, but the results are unsatisﬁed. Therefore, all these datasets with ‘NAN’ values were removed from the datasets.
Open, close, high and low price is used in the experiment. At ﬁrst, only the high and low price were used since high and low contains more information according to Siripurapu(Siripurapu, 2014). High and low price are bound of daily stock price, which also contains the open and close price. Open and close is in a sense of statistical artifacts, which are the prices sampled by the Google or Yahoo. However, after tried with only open and close price, the result was not good. Information on input images only with open and close price is not enough for the deep network to extract enough useful features. Therefore, in next experiment, four kinds of the price of the stock are input into the network, then the network can automatically choose which one is more useful.
Then images of stock price would be poled as RGB with size 5*5 kb. The windows size of an image is n 180, 90, 60, 30, which means using n days history data pictures to predict future, m30,20,10,5days. An example input image is showed as

<div align=center><img width="350" height="350" src="https://github.com/CharlesLoo/stockPrediction_CNN/blob/master/paper/input.PNG"/></div>

# Prediction strategy 
As said previous, prediction strategy is using previous n day’s history data pictures to predict future m days. 4 combination of n and m are used including using 180 days history data to predict 30 days price, using 90 days history to predict 20 days price, using 60 days history data to predict 10 days price and using 30 days history data to predict 5 days price.
These pictures would be labeled by the return of the price after m days. The return here is logarithmic return.
Price is mean of open, close, high and low price. ‘t’ is current time and ‘t+m’ is the future price. When ‘r’ > 0, it will be labeled ‘1’ which means ‘up’, otherwise it is ‘-1’.

<div align=center><img width="450" height="150" src="https://github.com/CharlesLoo/stockPrediction_CNN/blob/master/paper/fomular.PNG"/></div>

# Result
As expected, with much more layers network, VGG19 need much more time to train. Because this mode takes more memory, however, this model does have much better results than the Alexnet. As shown In Fig 4.12 it can achieve the smallest testing error of about 0.61 and with the best test accuracy of about 58%. Unfortunately,withthetraininggoeson,thetestlosswouldincreaseagain,sincethismodel isalsooverﬁtting. Butitisstillbetterthanmostpreviousworks,likePoulos(Poulos, 2014)’s 3-layer LSTM (55.45%) and SVM (56.31%) model.

<div align=center><img width="350" height="350" src="https://github.com/CharlesLoo/stockPrediction_CNN/blob/master/paper/loss2.PNG"/></div>
