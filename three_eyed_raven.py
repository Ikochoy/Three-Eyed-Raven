"""
Project Name: Three Eyed Raven
Author: Koby Choy
Date: 20th May, 2019
===============================================================================
Project Aim: I have built a program to analyze the stock data of 0700, Tencent
Holdings. This project aims to forecast the closing stock price of 0700 in the
next day.
===============================================================================
"""
import pandas as pd
import math
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import statistics

csv_file = open('tencent_data.csv')
df = pd.read_csv(csv_file)

df = df[['High', 'Low', 'Close', 'Trans Amount', 'HSI_idx', 'SSE_idx',
         'PE_idx', 'RSI_idx', 'ADR_Close_HK']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100
# defining a new variable call HL_PCT
df['PCT_change'] = (df['ADR_Close_HK'] - df['Close']) / df['Close'] * 100

df = df[['Close', 'HL_PCT', 'PCT_change', 'Trans Amount', 'HSI_idx',
         'SSE_idx', 'PE_idx', 'RSI_idx']]

forecast_col = 'Close'
df.fillna(0, inplace=True)

forecast_out = int(math.ceil(((1 / (len(df))) * len(df))))

df['labels'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['labels'], 1))
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['labels'])

results = []
for i in range(10000):
    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(x, y, test_size=0.2)
    # building a classifier
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    forecast_data = clf.predict(x_lately)
    results.append(float(forecast_data))

print("tmr:", str(statistics.median(results)))
