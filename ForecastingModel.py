import streamlit as st
import pandas as pd
import numpy as np

import pathlib
from abc import ABC, abstractmethod


PROJECT_DIR = pathlib.Path.cwd()
DATA_DIR = PROJECT_DIR / "data"


class ForecastingModel(ABC):
    def __init__(self, model, modelName, modelDescription):
        self.model = model
        self.modelName = modelName
        self.modelDescription = modelDescription

    @abstractmethod
    def fit(self, municipalityId):
        pass

    @abstractmethod
    def predict(self, municipalityId):
        pass

    @abstractmethod
    def plot(self, municipalityId):
        pass

    @abstractmethod
    def evaluate(self, municipalityId):
        pass

    def intro(self):
        st.markdown("### {}".format(self.modelName))
        st.markdown(self.modelDescription)

    def getModel(self):
        return self.model

    def getModelName(self):
        return self.modelName

    def getModelDescription(self):
        return self.modelDescription
    
    def mean_absolute_percentage_error(y_true, y_pred):
        """  Mean Absolute Percentage Error - MAPE """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
