import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data_swt_iee_usp_2018.csv", sep=";")

sc = StandardScaler()

def preprocess(df, scaling=False):    
    mapping_dict = {3: 2, 8: 3, 32: 4, 33: 5, 35: 6, 256: 7, 1280: 8}
    df['Turbine status'] = df['Turbine status'].replace(mapping_dict)
    y = df.loc[:, ['Turbine status']]
    split_df = df["Log Time"].str.split(',', expand=True)
    df.set_index(split_df[0], inplace=True)
    df = df.drop(["Grid status", "System status", "Slave Status", "Access Status",
                          "SN#", "Inv Time", "Software rev", "opversion", "Timer",
                          "voltage rise", "watt-hours", "Log Time", "Turbine status"], axis=1)

    if scaling == True:
        df = sc.transform(df)
        return df, y
    return df.values, y
        


df1, y1 = preprocess(df, scaling=False)
sc.fit_transform(df1)

df2, y2 = preprocess(df, scaling=True)
df3, y3 = preprocess(df, scaling=True)
df4, y4 = preprocess(df, scaling=True)
df5, y5 = preprocess(df, scaling=True)
df6, y6 = preprocess(df, scaling=True)
# =============================================================================
# df2 = df[20:40].values.reshape((1, 20, 26))
# df3 = df[40:60].values.reshape((1, 20, 26))
# df4 = df[60:80].values.reshape((1, 20, 26))
# df5 = df[80:100].values.reshape((1, 20, 26))
# 
# =============================================================================
model = Sequential()
model.add(LSTM(500, activation='relu', input_shape=(20, 26)))
#model.add(LSTM(100, return_sequences=False, activation='relu', input_shape=(n_steps, n_features)))
#model.add(LSTM(20, return_sequences=False, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model = tf.keras.models.load_model("bigData.h5")

test_stack = np.zeros(shape=(500, 20, 26))
times = []

val = 20
n_steps = 20
for idx in range(500):
    test_stack[idx] = np.vstack([df6[idx + x] for x in range(n_steps)])
    times.append(y6.index[val+idx])
times = np.array(times).reshape(-1, 1)
y_pred = model.predict(test_stack)
y_pred = np.argmax(y_pred, axis=1)
y_pred = y_pred.reshape(-1, 1)

final_ans = np.hstack((times, y_pred))
DF = pd.DataFrame(final_ans, columns=['Time', 'Predictions'])
 
# save the dataframe as a csv file
DF.to_csv("bigData_1.csv")