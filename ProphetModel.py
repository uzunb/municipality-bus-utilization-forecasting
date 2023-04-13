from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
from ForecastingModel import ForecastingModel

from prophet import Prophet
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
import numpy as np

import pathlib

PROJECT_DIR = pathlib.Path.cwd()
DATA_DIR = PROJECT_DIR / "data"


class ProphetModel(ForecastingModel):
    def __init__(self):
        super().__init__(model=Prophet(),
                         modelName="Prophet",
                         modelDescription="Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data.")

    def __loadData(self, municipalityId):
        self.df = pd.read_csv(DATA_DIR / 'municipality_bus_utilization.csv')
        self.df['timestamp'] = pd.to_datetime(
            self.df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        self.df = self.df.set_index('timestamp')

        self.df = self.df[self.df['municipality_id'] == municipalityId]
        self.df.drop(columns=['municipality_id'], inplace=True)

    def __preprocess(self):
        self.df['date'] = pd.to_datetime(self.df.index.date)
        self.df['time'] = self.df.index.time
        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month
        self.df['dayOfWeek'] = self.df.index.dayofweek
        self.df['day'] = self.df.index.day
        self.df['hour'] = self.df.index.hour
        self.df['minute'] = self.df.index.minute
        self.df['second'] = self.df.index.second
        self.df['quarter'] = self.df.index.quarter
        self.df['dayOfYear'] = self.df.index.dayofyear

        self.df['usage'] = self.df.apply(
            lambda x: x['total_capacity'] if x['usage'] > x['total_capacity'] else x['usage'], axis=1)
        self.df['usage_percentage'] = self.df['usage'] / \
            self.df['total_capacity']*100
        self.df.sort_values(by='usage_percentage', ascending=False).head()

        check_df = self.df[['date', 'hour']].groupby(
            ['date', 'hour']).size().reset_index(name='counts')
        check_df.sort_values('counts', ascending=True)

        new_timestamps = pd.DataFrame(columns=[
                                      'timestamp', 'usage', 'total_capacity', 'date', 'time', 'year', 'month', 'dayOfWeek', 'day', 'hour'])
        for date in self.df['date'].unique():
            for hour in self.df['hour'].unique():
                if check_df[(check_df['date'] == date) & (check_df['hour'] == hour)]['counts'].values == 1:
                    date = pd.to_datetime(date)
                    new_record = {'timestamp': f"{date} {hour}:00:00", 'usage': np.nan, 'total_capacity': np.nan,
                                  'date': date, 'time': datetime.strptime(f"{hour}:00:00", '%H:%M:%S').time(), 'year': date.year, 'month': date.month, 'dayOfWeek': date.dayofweek, 'day': date.day, 'hour': hour,
                                  'minute': 0, 'second': 0, 'quarter': date.quarter, 'dayOfYear': date.dayofyear, 'usage_percentage': np.nan}
                    new_timestamps = pd.concat(
                        [new_timestamps, pd.DataFrame.from_records([new_record])])

        # add new timestamp to df
        expanded_df = self.df.copy()
        expanded_df = expanded_df.reset_index()
        expanded_df = pd.concat(
            [expanded_df, new_timestamps], ignore_index=True)
        expanded_df['timestamp'] = pd.to_datetime(expanded_df['timestamp'])
        expanded_df = expanded_df.set_index('timestamp')
        expanded_df.sort_index(inplace=True)

        # impute missing values for each municipality
        imputed_df = expanded_df.copy()

        # fill total_capacity with max value
        imputed_df['total_capacity'] = imputed_df['total_capacity'].fillna(
            imputed_df['total_capacity'].max())

        # fill usage with interpolation that is best appropriate of method
        imputed_df['usage'] = imputed_df['usage'].fillna(method='bfill')
        imputed_df['usage'] = imputed_df['usage'].fillna(method='ffill')

        imputed_df['usage_percentage'] = imputed_df['usage'] / \
            imputed_df['total_capacity']*100

        self.df = imputed_df.copy()

    def fit(self, municipalityId):
        self.__loadData(municipalityId=municipalityId)
        self.__preprocess()

        FEATURES = self.df.drop(columns=['usage', 'time']).columns
        TARGET = 'usage'

        # split data to train and test set
        horizon = pd.to_datetime("2017-08-05")

        train = self.df[self.df.date < horizon]
        test = self.df[self.df.date >= horizon]

        # prepare data for training
        train = train.reset_index()
        train['ds'] = train['timestamp']
        train['y'] = train['usage']

        test = test.reset_index()
        test['ds'] = test['timestamp']
        test['y'] = test['usage']

        # change data type of columns
        train['total_capacity'] = train['total_capacity'].astype('int')
        train['usage_percentage'] = train['usage_percentage'].astype('float64')
        train['y'] = train['y'].astype('int')

        test['total_capacity'] = test['total_capacity'].astype('int')
        test['usage_percentage'] = test['usage_percentage'].astype('float64')
        test['y'] = test['y'].astype('int')

        train.drop(columns=['timestamp'], inplace=True)
        train.drop(columns=['usage'], inplace=True)
        train.drop(columns=['time'], inplace=True)
        test.drop(columns=['timestamp'], inplace=True)
        test.drop(columns=['usage'], inplace=True)
        test.drop(columns=['time'], inplace=True)

        self.train_df = train.copy()
        self.test_df = test.copy()

        # add regressors
        for feature in FEATURES:
            self.model.add_regressor(feature)

        # fit model
        self.model.fit(train)

    def predict(self):
        future = self.model.make_future_dataframe(
            periods=self.test_df.shape[0])
        future = future.merge(self.test_df, on='ds', how='right')
        forecast = self.model.predict(future)

        return forecast

    def plot(self, municipalityId):
        forecast = self.predict()
        forecast['y'] = self.test_df['y'].values
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']]
        forecast = forecast.rename(columns={'ds': 'timestamp', 'yhat': 'prediction',
                                   'yhat_lower': 'prediction_lower', 'yhat_upper': 'prediction_upper', 'y': 'actual'})

        fig = go.Figure()

        # Add training data
        fig.add_trace(go.Scatter(
            x=self.train_df['ds'], y=self.train_df['y'], name='Training'))

        # Add testing data
        fig.add_trace(go.Scatter(
            x=self.test_df['ds'], y=self.test_df['y'], name='Testing'))

        # Add predicted values
        fig.add_trace(go.Scatter(
            x=forecast['timestamp'], y=forecast['prediction'], name='Predicted'))

        # Add upper and lower bounds
        fig.add_trace(go.Scatter(
            x=forecast['timestamp'], y=forecast['prediction_upper'], name='Upper Bound'))
        fig.add_trace(go.Scatter(
            x=forecast['timestamp'], y=forecast['prediction_lower'], name='Lower Bound'))

        fig.update_layout(title='Forecast vs Actuals for {}'.format(municipalityId),
                          xaxis_title='Date', yaxis_title='Usage')

        st.plotly_chart(fig)

    def evaluate(self, municipalityId):
        forecast = self.predict()
        forecast['y'] = self.test_df['y'].values
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']]
        forecast = forecast.rename(columns={'ds': 'timestamp', 'yhat': 'prediction',
                                   'yhat_lower': 'prediction_lower', 'yhat_upper': 'prediction_upper', 'y': 'actual'})
        forecast = forecast.set_index('timestamp')

        # calculate MAPE
        mape = self.mean_absolute_percentage_error(
            forecast['actual'], forecast['prediction']
        )

        # calculate MAE
        mae = mean_absolute_error(forecast['actual'], forecast['prediction'])

        # calculate MSE
        mse = mean_squared_error(forecast['actual'], forecast['prediction'])

        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(
            forecast['actual'], forecast['prediction']))

        # plot metrics in plotly
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=['MAPE', 'MAE', 'MSE', 'RMSE'], y=[mape, mae, mse, rmse]))
        fig.update_layout(title='Evaluation Metrics for {}'.format(municipalityId),
                          xaxis_title='Metric', yaxis_title='Value')
        st.plotly_chart(fig)
