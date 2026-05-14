# Generated from: eda.ipynb
# Converted at: 2026-05-14T12:43:11.003Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd

df = pd.read_excel("merged_cleaned_retail_data.xlsx")

print(df.head())

print(df.shape)

print(df.isnull().sum())

print(df.info())

print(df.describe())

import matplotlib.pyplot as plt

df.hist(figsize=(15,10))

plt.show()

import seaborn as sns

plt.figure(figsize=(10,8))

sns.heatmap(
    df.select_dtypes(include='number').corr(),
    annot=True,
    cmap='coolwarm'
)

plt.show()

df = df.dropna()

df = df.drop_duplicates()

df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])

df['TotalAmount'] = df['Quantity'] * df['Price']

snapshot_date = df['Invoice Date'].max()

rfm = df.groupby('Customer ID').agg({
    'Invoice Date': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalAmount': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

print(rfm.head())

daily_sales = df.groupby(
    df['Invoice Date'].dt.date
)['TotalAmount'].sum()

rolling_mean = daily_sales.rolling(7).mean()

print(rolling_mean.head())

!pip install great_expectations==0.18.21
!pip install pydantic==1.10.13

!pip uninstall great_expectations -y
!pip uninstall pydantic -y

!pip cache purge

!pip install great_expectations==0.18.21
!pip install pydantic==1.10.13

import great_expectations as gx

context = gx.get_context()

print("Validation Ready")

!pip install scikit-learn

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    rfm[['Recency', 'Frequency', 'Monetary']]
)

from scipy.cluster.vq import kmeans2

centroids, labels = kmeans2(
    scaled_data,
    4,
    minit='points'
)

rfm['Cluster'] = labels

print(rfm.head())

import matplotlib.pyplot as plt

plt.scatter(
    rfm['Frequency'],
    rfm['Monetary'],
    c=rfm['Cluster']
)

plt.xlabel("Frequency")

plt.ylabel("Monetary")

plt.show()

daily_sales = df.groupby(
    df['Invoice Date'].dt.date
)['TotalAmount'].sum()

daily_sales.index = pd.to_datetime(
    daily_sales.index
)

from statsmodels.tsa.stattools import adfuller

result = adfuller(daily_sales)

print("ADF Statistic:", result[0])

print("P-value:", result[1])

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(
    daily_sales,
    model='additive',
    period=30
)

decomposition.plot()

!pip install prophet

from prophet import Prophet

prophet_df = daily_sales.reset_index()

prophet_df.columns = ['ds', 'y']

model = Prophet()

model.fit(prophet_df)

future = model.make_future_dataframe(
    periods=30
)

forecast = model.predict(future)

model.plot(forecast)

!pip install torch pytorch-lightning

import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        return self.fc(out[:, -1, :])

model = LSTMModel()

print(model)

df.to_csv(
    "cleaned_dataset.csv",
    index=False
)

rfm.to_csv(
    "customer_segmentation.csv"
)