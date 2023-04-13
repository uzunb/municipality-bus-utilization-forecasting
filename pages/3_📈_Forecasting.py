import streamlit as st
import pandas as pd
import numpy as np

import pathlib

from ForecastingModel import ForecastingModel
from ProphetModel import ProphetModel
from XGBoostModel import XGBoostModel

PROJECT_DIR = pathlib.Path.cwd()
DATA_DIR = PROJECT_DIR / "data"



def app():
    st.set_page_config(page_title="Forecasting", page_icon="📈")

    st.markdown("""
    # MUNICIPALITY BUS UTILIZATION FORECASTING

    You can explore and forecast the bus demand of municipalities in Banana Republic using the dataset.
    """)
    st.markdown("## Overview")

    municipality_id = st.sidebar.selectbox(
        label="Municipality ID",
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        index=0
    )

    # plot the data for the selected municipality
    df = pd.read_csv(DATA_DIR / "municipality_bus_utilization.csv")
    df = df[df["municipality_id"] == municipality_id]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    st.line_chart(df[["usage", "total_capacity"]])

    st.markdown("## Forecasting")
    model: ForecastingModel = st.sidebar.selectbox(
        label="Model",
        options=["Prophet", "XGBoost"],
        index=0
    )

    if model == "Prophet":
        model = ProphetModel()
    elif model == "XGBoost":
        model = XGBoostModel()
        

    model.intro()
    model.fit(municipalityId=municipality_id)
    model.plot()
    model.evaluate()
    # model.forecast()



app()