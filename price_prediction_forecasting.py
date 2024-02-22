# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
import matplotlib.pyplot as plt

train_file = '/kaggle/input/CiVilium/train_final.csv'
test_file = '/kaggle/input/CiVilium/test_final.csv'

raw_train_data = pd.read_csv(train_file)
raw_test_data = pd.read_csv(test_file)

# Examine train file
print("Train data info():")
print(raw_train_data.info())
print("================")
print('Description of each column')
raw_train_data.describe().transpose()

# Check the distribution of steps in timestamps in train data
diff_train_timestamp = raw_train_data['Unix_Timestamp'].diff()
plt.hist(diff_train_timestamp[1:],bins=list(range(0,1800,100)))

# Check if there is any invalid value in each row in train set
print("Number of negative transaction values in Volume_CiVilium: {}".format(raw_train_data[raw_train_data['Volume_CiVilium'] < 0]['Volume_CiVilium'].size))
print("Number of negative Average Price in Weighted_Price: {}".format(raw_train_data[raw_train_data['Weighted_Price']<0]['Weighted_Price'].size))
print("Number of NaN in Volume_CiVilium: {}".format(raw_train_data['Volume_CiVilium'].isnull().sum()))
print("Number of NaN in Weighted_Price: {}".format(raw_train_data['Weighted_Price'].isnull().sum()))

# A few rows in the train data
raw_train_data.head(10)

# Plot the historical data of volume and weighted price 
fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(raw_train_data['Unix_Timestamp'],raw_train_data['Volume_CiVilium'])
ax1.set_title('Volume of CiVilium')
ax1.set_xlabel('timestamp')
ax1.set_ylabel('Units (Uts)')

ax2.plot(raw_train_data['Unix_Timestamp'],raw_train_data['Weighted_Price'])
ax2.set_title('Volume Weighted Average Price')
ax2.set_xlabel('timestamp')
ax2.set_ylabel('Zibabwean Dollar (ZWD)')

fig.tight_layout()
plt.show()

# add a new column 
train_data_new_col = raw_train_data.assign(Total_Amount=raw_train_data['Volume_CiVilium']*raw_train_data['Weighted_Price'])
train_data_new_col.sample(10)

# A few raws of the new dataframe with the new col 'Total_Amount'
plt.plot(train_data_new_col['Unix_Timestamp'],train_data_new_col['Total_Amount'])
plt.xlabel('timestamp')
plt.ylabel('Zibabwean Dollar (ZWD)')
plt.title('Total Amount')

# Check if there is any invalid value in each row in test set
print("Number of negative transaction values in Volume_CiVilium: {}".format(raw_test_data[raw_test_data['Volume_CiVilium'] < 0]['Volume_CiVilium'].size))
print("Number of NaN in Volume_CiVilium: {}".format(raw_test_data['Volume_CiVilium'].isnull().sum()))

# Check the distribution of timestamps diff in test data
diff_test_timestamps = raw_test_data['Unix_Timestamp'].diff()
plt.hist(diff_test_timestamps,bins=list(range(0,2000,100)))

# ref https://stats.stackexchange.com/questions/368863/time-series-regression-non-contiguous-dataset
# ref: https://inventoro.com/sales-forecasting-for-intermittent-or-sporadic-demand/
# ref: https://demand-planning.com/2009/10/08/understanding-intermittent-demand-forecasting-solutions/
# ref: https://stats.stackexchange.com/questions/74538/time-series-analysis-for-non-uniform-data
# ref: https://www.linkedin.com/pulse/forecasting-intermittent-demand-new-approach-levenbach-phd-cpdf/

# excluding outliers in volume (3 or more SD away from mean)
raw_train_data = raw_train_data[raw_train_data['Volume_CiVilium'] <= (raw_train_data['Volume_CiVilium'].mean() + raw_train_data['Volume_CiVilium'].std())]
raw_train_data = raw_train_data[raw_train_data['Volume_CiVilium'] >= (raw_train_data['Volume_CiVilium'].mean() - raw_train_data['Volume_CiVilium'].std())]
raw_train_data.reset_index(drop=True)
raw_train_data.describe().transpose()

# plot time dependene of Volume_CiVilium again after the removal of outliers
plt.plot(raw_train_data['Unix_Timestamp'],raw_train_data['Volume_CiVilium'])
plt.title('Volume_CiVilium')
plt.xlabel('timestamp')
plt.ylabel('Units (Uts)')
plt.show()

# Handling discontiguous time series data as the delta between timestamps varies among dataset
# ref: https://machinelearningmastery.com/faq/single-faq/how-do-i-handle-discontiguous-time-series-data/
# ref: https://machinelearningmastery.com/taxonomy-of-time-series-forecasting-problems/

# Global variables
from dataclasses import dataclass

raw_train_data_count = len(raw_train_data)
@dataclass
class G:
    split_ratio = 0.8
    split_num = int(raw_train_data_count*split_ratio)
    time_steps = 1
    features = 2
    shuffle_buffer_size = 5000
    batch_size = 256
    lstm_dims = 128
    output_dims = 1
    epochs = 1500
    learning_rate = 1e-7 #1e-6
    patience = 3

# Standardize the column of timestamp 
def timestamp_standardization(series,base):
    if base == 0:
        return series
    return [[item[0]/base,item[1]] for item in series]

# Multivariate LSTM 
# ref: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
train_data = raw_train_data[:G.split_num]
valid_data = raw_train_data[G.split_num:]
# split data input and output
train_data_x, train_data_y = train_data.values[:,:-1], train_data.values[:,-1]
valid_data_x, valid_data_y = valid_data.values[:,:-1], valid_data.values[:,-1]
# standardization
beginning_timestamp = train_data_x[0][0]
train_data_x = timestamp_standardization(train_data_x,beginning_timestamp)
valid_data_x = timestamp_standardization(valid_data_x,beginning_timestamp)

# Convert list to TensorFlow dataset
train_data_single_step = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(train_data_x),
    tf.data.Dataset.from_tensor_slices(train_data_y)
))
valid_data_single_step = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(valid_data_x),
    tf.data.Dataset.from_tensor_slices(valid_data_y)
))

# shuffle data
train_data_single_step = train_data_single_step.shuffle(G.shuffle_buffer_size)

# function for preprocessing data, including batch
def data_format(features,price):
    features = tf.expand_dims(features,0)
    return features,price

# Prepare data for LSTM's input
train_input_single_step = train_data_single_step.map(data_format)
valid_input_single_step = valid_data_single_step.map(data_format)

# batch data
train_input_single_step = train_input_single_step.batch(G.batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
valid_input_single_step = valid_input_single_step.batch(G.batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Visual inspection of the batched data and verify shape of the tensor is [samples, timesteps, features]
# ref: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# ref: https://stackoverflow.com/questions/39324520/understanding-tensorflow-lstm-input-shape
for d in train_input_single_step.take(1):
    print(d)

# ML Model using LSTM
# ref: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
model_single_step = tf.keras.Sequential([
    tf.keras.layers.LSTM(G.lstm_dims,input_shape=(G.time_steps,G.features),return_sequences=True),
    tf.keras.layers.LSTM(int(G.lstm_dims),return_sequences=True),
    tf.keras.layers.LSTM(int(G.lstm_dims/2),return_sequences=True),
    tf.keras.layers.LSTM(int(G.lstm_dims/2)),
    tf.keras.layers.Dense(G.lstm_dims),
    tf.keras.layers.Dense(G.output_dims,activation='relu')
])

model_single_step.summary()

callback_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_mae',
    patience = G.patience
)

history_single_step = model_single_step.fit(
    train_input_single_step,
    validation_data=valid_input_single_step,
    epochs=G.epochs,
    callbacks=[callback_earlystop]
)

history_params = history_single_step.history
history_params.keys()

# parameters for single-step
train_mae = history_params['mae']
val_mae = history_params['val_mae']
epochs = range(1,len(val_mae)+1)
# plot training history
plt.plot(epochs,train_mae,'bo',label='Train MAE')
plt.plot(epochs,val_mae,'b',label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Predict test data
test_data = timestamp_standardization(raw_test_data.values,beginning_timestamp)
test_data = tf.data.Dataset.from_tensor_slices(test_data)
test_data = test_data.map(lambda x: tf.expand_dims(x,0))
test_data = test_data.batch(G.batch_size).cache().prefetch(buffer_size = tf.data.AUTOTUNE)
for t in test_data.take(1):
    print(t)

# predict prices
result_single_step = model_single_step.predict(
    test_data,
    batch_size = G.batch_size
)
test_timestamps = [t[0] for t in raw_test_data.values]
test_result_single_step = [r[0] for r in result_single_step]
new_df = {'Unix_Timestamp': test_timestamps,'Weighted_Price': test_result_single_step}
output = pd.DataFrame(new_df)
col_type = {'Unix_Timestamp': int, 'Weighted_Price': float}
output.astype(col_type)
output = output.to_numpy()
output
