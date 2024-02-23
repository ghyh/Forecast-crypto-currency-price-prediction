# TimeSeriesForecasting-crypto-currency-price-prediction
## Introduction
The ML model in this repo is to study the Kaggle competition, "CiVilium Price Prediction" (https://www.kaggle.com/competitions/CiVilium/overview)[1], to predict the future trends of the price of a hypothetical cryptocurrency, CiVilium, based on the data of historical prices. 
## Data Wrangling and Feature Engineering
The provided data is split into train and test data sets. The training set has three columns, Unix_Timestamp, Volume_CiVilium and Weighted_Price while the test data set only has the first two. Examples can be found below.
![Screenshot 2023-12-17 at 17-29-19 PricePrediction Kaggle](https://github.com/ghyh/TimeSeriesForecasting-crypto-currency-price-prediction/assets/30448897/44c46705-e145-45f7-8e26-32e50cc24362)   
Basic check on training data and test data has been performed to verfiy that there is no invalid data points, such as negative or null data entry.
![Screenshot 2023-12-17 at 17-45-55 PricePrediction Kaggle](https://github.com/ghyh/TimeSeriesForecasting-crypto-currency-price-prediction/assets/30448897/b821428e-2944-4d4a-bb01-9bd922f75dd9)   
![Screenshot 2023-12-17 at 17-46-19 PricePrediction Kaggle](https://github.com/ghyh/TimeSeriesForecasting-crypto-currency-price-prediction/assets/30448897/5f872ec2-2c04-454b-90b9-59fa35c34d36)
Before feeding training data into the ML model, outliners were identified in the chart of historical data and basic statistics table
![Screenshot 2023-12-17 at 17-50-40 PricePrediction Kaggle](https://github.com/ghyh/TimeSeriesForecasting-crypto-currency-price-prediction/assets/30448897/2ccacc4a-e9fd-4f38-862f-ac5a4ae0d639)   
![Screenshot 2023-12-17 at 17-56-11 PricePrediction Kaggle](https://github.com/ghyh/TimeSeriesForecasting-crypto-currency-price-prediction/assets/30448897/2ca1257d-bd2a-4234-8bef-b53249ab01d6)   
### Exclusion of Outliers
After excluding outlier data points in the column "Volume_CiVilium" that are 3 or more SD away from the mean, the fluctuation in the chart of historical data on Volume_CiVilium has been significantly reduced, as shown below. The SD has been reduced from 48 to 11.
![Screenshot 2023-12-17 at 17-59-53 PricePrediction Kaggle](https://github.com/ghyh/TimeSeriesForecasting-crypto-currency-price-prediction/assets/30448897/c2e1fdd5-f4ae-4e5a-b8ea-4373916f7bf3)   
### Nomralization of Timestamp Column
The train data set is split into two subsets, where the first 80% of data points in the timeline are training data while the rest of 20% data points are the validation data. The values of timestamp in the above two sets have been normalized using the first (i.e. the earliest) timestamp to reduce the over-presentation of the column "Unix_Timestamp" in the training process.
## Machine Learning Model
The ML model used in this study is a neural network composed of (1) 4 LSTM layers and (2) 2 Dense layers, as shown below.
![Screenshot 2023-12-17 at 18-15-54 PricePrediction Kaggle](https://github.com/ghyh/TimeSeriesForecasting-crypto-currency-price-prediction/assets/30448897/3baff3ff-2e5e-452e-96a3-4ceb2fa3b671)   
After testing with a single layer of LSTM (single step test), 4 layers of LTSM layers have been chosen for the best results of prediction. In the Dense layer at the output, ReLu (Rectified Linear Unit)[8],is used as activiation function. The hyperparameters in this study, such as the dimension of the LSTM model (lstm_dims),size of the training batch (batch_size), epochs (epochs) and learning rate (learning_rate), were chosen after carefully calculating the loss of validation. 
## Result and Discussion
The mean absolute error (MAE) is used as the loss for comparison between training and validation data. The results, including the table of caculated MAE and its trend, can be found below.
One can see that as more epochs goes, the accuracy for both training and validation data sets keeps increasing, and the MAE continues goes downward. This is a good sign that we are not overfitting the training data.   
||Training Set|Validation Set|
|---|---|---|
|MAE|1.3483|0.2940|   

![Screenshot 2024-02-23 at 00-31-14 PricePrediction Kaggle](https://github.com/ghyh/TimeSeriesForecasting-crypto-currency-price-prediction/assets/30448897/0058c19c-f328-4e92-905b-601e3cfe94fd)   
When being applied to the test data set, the model also generated good result and finished 6th place on Kaggle's leaderboard. The effort of building a forecasting model using other ML algorithms to compare with the current result is ongoing. 
### Reference
1. CiVilium Price Prediction, Kaggle https://www.kaggle.com/competitions/CiVilium/overview   
2. Multivariate Time Series Forecasting with LSTMs in Keras, Machine Learning Mastery https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/   
3. How to Develop a Skillful Machine Learning Time Series Forecasting Model, Machine Learning Mastery https://machinelearningmastery.com/how-to-develop-a-skilful-time-series-forecasting-model/   
4. Long Short-term Memory,Hochreiter & Schmidhuber, 1997 https://www.bioinf.jku.at/publications/older/2604.pdf   
5. Using a Simple RNN for forecasting, Repository "https-deeplearning-ai/tensorflow-1-public/C4/W3/ungraded_labs" in deeplearning.ai,https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/64287d2983de3ffd6d6de9c26d9f1f558fcd3968/C4/W3/ungraded_labs/C4_W3_Lab_1_RNN.ipynb   
6. How can Tensorflow be used to standardize the data using Python?, GeeksforGeeks, https://www.geeksforgeeks.org/how-can-tensorflow-be-used-to-standardize-the-data-using-python/   
7. Multivariate Time Series Forecasting Using Random Forest, Hafidz Zulkifli, Medium https://towardsdatascience.com/multivariate-time-series-forecasting-using-random-forest-2372f3ecbad1
8. A Gentle Introduction to the Rectified Linear Unit (ReLU), Machine Learning Mastery https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/   
