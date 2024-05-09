#Import libraries
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import pickle

#Read data
scada_df = pd.read_csv("data_swt_iee_usp_2018.csv", sep=";")

#Drop rows that can leak data
scada_df = scada_df.drop(["Grid status", "System status", "Slave Status", "Access Status",
                          "SN#", "Inv Time", "Software rev", "opversion", "Timer",
                          "voltage rise", "watt-hours"], axis=1)


mapping_dict = {3: 2, 8: 3, 32: 4, 33: 5, 35: 6, 256: 7, 1280: 8}

scada_df['Turbine status'] = scada_df['Turbine status'].replace(mapping_dict)
#print(scada_df["Log Time"].isna().sum())

split_df = scada_df["Log Time"].str.split(',', expand=True)

scada_df.set_index(split_df[0], inplace=True)
scada_df.drop(columns=['Log Time'], inplace=True)


train_size = int(len(scada_df) * 0.8)
train, test = scada_df.iloc[:train_size], scada_df.iloc[train_size:]

Xtr = train.loc[:, ~scada_df.columns.isin(['Turbine status'])].values
ytr = train.loc[:, ['Turbine status']]

Xts = test.loc[:, ~scada_df.columns.isin(['Turbine status'])].values
yts = test.loc[:, ['Turbine status']]

sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xts = sc.transform(Xts)

n_steps = 20 #10 steps equal 10 minutes approx
n_features = 26

stack = np.zeros(shape=(len(Xtr) - n_steps, n_steps, n_features))

for idx in range(len(Xtr) - n_steps - 1):
    stack[idx] = np.vstack([Xtr[idx + x] for x in range(n_steps)])
    
ytr = ytr[n_steps:]
for i in np.unique(ytr):
    print(i)

model = Sequential()
model.add(LSTM(500, activation='relu', input_shape=(n_steps, n_features)))
#model.add(LSTM(100, return_sequences=False, activation='relu', input_shape=(n_steps, n_features)))
#model.add(LSTM(20, return_sequences=False, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import Callback

class NaNDebugger(Callback):
    def on_batch_end(self, batch, logs=None):
        if logs.get('loss') is not None and np.isnan(logs.get('loss')):
            print(f'Batch {batch}: Invalid loss, terminating training')
            self.model.stop_training = True

model.fit(stack, ytr, epochs=2, callbacks=[NaNDebugger()])


#model.fit(stack, ytr, epochs=20, verbose=1)

test_stack = np.zeros(shape=(500, n_steps, n_features))
times = []

val = n_steps
for idx in range(500):
    test_stack[idx] = np.vstack([Xts[idx + x] for x in range(n_steps)])
    times.append(yts.index[val+idx])
times = np.array(times).reshape(-1, 1)
y_pred = model.predict(test_stack)

y_pred = np.argmax(y_pred, axis=1)
y_pred = y_pred.reshape(-1, 1)

final_ans = np.hstack((times, y_pred))

model.save("bigData.h5")

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)