import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# CSVファイルの列名を明示的に指定し、不要な先頭3行を読み飛ばします。
col_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
dataset_train = pd.read_csv('Tesla_Stock_Price_Train.csv', skiprows=3, names=col_names)
training_set = dataset_train[['Open']].values

# 特徴量のスケーリング
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# 60タイムステップを入力、1つの値を出力とするデータ構造を作成
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# データ形状の変更
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# パート2 - RNNモデルの構築

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# （注）学習時間を短縮するため、epochsを10に設定しています。必要に応じて100に増やしてください
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)


# パート3 - 予測と結果の可視化

# テストデータセットも同じ方法で読み込みます。
dataset_test = pd.read_csv('Tesla_Stock_Price_Test.csv', skiprows=3, names=col_names)
real_stock_price = dataset_test[['Open']].values

# 予測用データの準備
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 60 + len(dataset_test)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# 【修正済み】結果の可視化で日付をX軸に使用
# 1. 可視化用にテストデータの日付を取得します。
test_dates = pd.to_datetime(dataset_test['Date'])

# 2. 結果を可視化し、plt.plot()のX軸に日付データを指定します。
plt.figure(figsize=(14, 7))
plt.plot(test_dates, real_stock_price, color = 'red', label = 'Real Tesla Stock Price')
plt.plot(test_dates, predicted_stock_price, color = 'blue', label = 'Predicted Tesla Stock Price')
plt.title('Tesla Stock Price Prediction')
plt.xlabel('Date') # 3. X軸ラベルを「Date」に変更します
plt.ylabel('Tesla Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()