# MUNICIPALITY BUS UTILIZATION FORECASTING

Forecasting bus demand in Banana Republic municipalities.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uzunb-municipality-bus-utilization-forecas-1--enter-page-96l8kc.streamlit.app/)

## About

### Description

The central urban planning commitee of Banana Republic asked you to help them with the forecast of bus demands of municipalities.

### Dataset

They provide a nice dataset to support you (<https://pi.works/3w8IJbV>). The dataset includes two measurements for an hour for the number of used buses in each municipality, each measurement is timestamped. The dataset format is as follows (comma separated values):
MUNICIPALITY_ID, TIMESTAMP, USAGE, TOTAL_CAPACITY
where municipality_id is an anonymization to disguise the actual names, timestamp represents the exact time of the measurement, usage is the number of buses in use at the time of measurement and total_capacity represents the total number of buses in the municipality. There are 10 municipalities (ids from 0 to 9), and two measurements for an hour.

### Task

The committee asks you to forecast the hourly bus usages for next week for each municipality. Hence you can aggregate the two measurements for an hour by taking the max value (sum would not be a nice idea for the obvious reasons) for each hour, and you should model this data with a time series model of your selection. (It would be a nice idea to implement a very simple baseline model first, and then try to improve the accuracy by introducing more complex methods eventually. The bare minimum requirement of the task is one simple baseline and one complex method.)

### Evaluation

The committee says that they will use the last two weeks (starting from 2017-08-05 to 2017-08-19) as assessment (test) data, hence your code should report the error (in the criterion you chose for the task) for the last two weeks. You may use true values for the prediction of the last week of test data, then combine the error of the first and last week of the test separately.

### Notes

Keep in mind that the dataset has missing data, hence a suitable missing data interpolation would be useful.

## Installation

### Requirements

- Python 3.8 or higher
- Pandas
- Numpy
- Matplotlib
- Seaborn

### Instructions

```bash
# 1. Clone the repository
git clone https://github.com/uzunb/municipality-bus-utilization-forecasting.git

# 2. Change the working directory
cd municipality-bus-utilization-forecasting

# 3. Create a virtual environment 
python3 -m virtualenv venv

# 4. Activate the virtual environment
source venv/bin/activate

# 5. Install the requirements
pip install -r requirements.txt

# 6. Run the code
python main.py
```
