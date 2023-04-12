from ForecastingModel import ForecastingModel

from prophet import Prophet
import streamlit as st
import pandas as pd
import numpy as np

import pathlib
from abc import ABC, abstractmethod


PROJECT_DIR = pathlib.Path.cwd()
DATA_DIR = PROJECT_DIR / "data"

class XGBoostModel(ForecastingModel):
    def __init__(self):
        super().__init__(model=None, 
                         modelName="XGBoost", 
                         modelDescription="XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.")

    def fit(self, municipalityId):
        pass

    def predict(self, municipalityId):
        pass

    def plot(self, municipalityId):
        pass

    def forecast(self, municipalityId):
        pass