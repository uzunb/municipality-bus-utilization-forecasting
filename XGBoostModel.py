from datetime import datetime
import pathlib
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
import numpy as np

from ForecastingModel import ForecastingModel

PROJECT_DIR = pathlib.Path.cwd()
DATA_DIR = PROJECT_DIR / "data"

class XGBoostModel(ForecastingModel):
    def __init__(self):
        super().__init__(model=None, 
                         modelName="XGBoost", 
                         modelDescription="XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.")
        
        self.X_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_test: pd.DataFrame = None

    def __loadData(self, municipalityId):
        # Read the data
        self.df = pd.read_csv(DATA_DIR / 'municipality_bus_utilization.csv')
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='%Y-%m-%d %H:%M:%S')
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

        # convert object to int
        self.df['year'] = self.df['year'].astype(int)
        self.df['month'] = self.df['month'].astype(int)
        self.df['dayOfWeek'] = self.df['dayOfWeek'].astype(int)
        self.df['day'] = self.df['day'].astype(int)
        self.df['hour'] = self.df['hour'].astype(int)
        self.df['minute'] = self.df['minute'].astype(int)
        self.df['second'] = self.df['second'].astype(int)
        self.df['quarter'] = self.df['quarter'].astype(int)
        self.df['dayOfYear'] = self.df['dayOfYear'].astype(int)

    def fit(self, municipalityId):
        self.__loadData(municipalityId=municipalityId)
        self.__preprocess()

        # split data to train and test set
        horizon = pd.to_datetime("2017-08-05")

        self.train_df = self.df[self.df.date < horizon]
        self.test_df = self.df[self.df.date >= horizon]

        FEATURES = self.df.drop(columns=['usage', 'time', 'date']).columns
        TARGET = 'usage'

        self.X_train = self.train_df[FEATURES]
        self.y_train = self.train_df[TARGET]

        self.X_test = self.test_df[FEATURES]
        self.y_test = self.test_df[TARGET]

        self.model = xgb.XGBRegressor(n_estimators=10000, learning_rate=0.001, n_jobs=7, random_state=42)
        self.model.fit(self.X_train, self.y_train,
                    early_stopping_rounds=50,
                    eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                    verbose=100)

    def predict(self):
        self.test_df['prediction'] = self.model.predict(self.X_test)
        self.df = self.df.merge(self.test_df[['prediction']], left_index=True, right_index=True, how='left')

    def plot(self, municipalityId):
        self.predict()

        # usage and prediction plot in plotly interactive
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['usage'], name='usage'))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['prediction'], name='prediction'))
        fig.update_layout(title=f"municipality_id: {municipalityId}", xaxis_title='timestamp', yaxis_title='usage and prediction', colorway=['#5e0dac', '#ffa600'])
        
        st.plotly_chart(fig)

    def evaluate(self, municipalityId):
        # calculate error metrics and plot in plotly interactive
        mae = mean_absolute_error(self.y_test, self.df['prediction'].loc[self.y_test.index])
        mse = mean_squared_error(self.y_test, self.df['prediction'].loc[self.y_test.index])
        rmse = np.sqrt(mse)
        mape = self.mean_absolute_percentage_error(self.y_test, self.df['prediction'].loc[self.y_test.index])

        fig = go.Figure()
        fig.add_trace(go.Bar(x=['mae', 'mse', 'rmse', 'mape'], y=[mae, mse, rmse, mape], name='error metrics'))
        fig.update_layout(title=f"municipality_id: {municipalityId}", xaxis_title='error metrics', yaxis_title='value')

        st.plotly_chart(fig)

        