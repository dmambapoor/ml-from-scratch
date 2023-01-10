import pandas as pd
import numpy as np
from pickle import dump
from ml_from_scratch.MLP import MLP

# Read in the dataset
austin_wdata = pd.read_csv("./examples/austin_weather_model/noaa_austin_2000_present.csv", index_col="DATE", usecols=["DATE", "TMIN", "TMAX", "PRCP"], dtype={"TMIN" : np.float64, "TMAX" : np.float64, "PRCP" : np.float64})

# Note: This data-set does not have null values for TMIN, TMAX.
# run the folllowing commands to verify:
#   print(austin_wdata.apply(pd.isnull).sum())
#   print(austin_wdata.apply(lambda x : (x==9999)).sum()) # '9999' indicates NOAA did not have temperature data for that field.

# Squash data within 0 to 1
for col in austin_wdata.columns:
    min_point = austin_wdata[col].min()
    max_point = austin_wdata[col].max()
    # Linear squash
    austin_wdata[col] = austin_wdata[col].apply(lambda x : (x - min_point)/(max_point-min_point))

# Add previous [window_size] days of data to each entry
window_size = 30
for col in austin_wdata.columns:
    for i in range(window_size):
        austin_wdata["%s_%i" % (col, i+1)] = austin_wdata.shift((i+1))["TMAX"]

# Add target
austin_wdata["TARGETMAX"] = austin_wdata.shift(-1)["TMAX"]

#Trim rows with not enough previous info to make a weekly window
austin_wdata = austin_wdata.iloc[window_size+1:-1, :].copy()
# print(austin_wdata)

x_train = austin_wdata.iloc[:int(austin_wdata.shape[0] * 0.8), :-1].to_numpy()
y_train = austin_wdata.iloc[:int(austin_wdata.shape[0] * 0.8), -1].to_numpy()
x_test = austin_wdata.iloc[int(austin_wdata.shape[0] * 0.8):, :-1].to_numpy()
y_test = austin_wdata.iloc[int(austin_wdata.shape[0] * 0.8):, -1].to_numpy()

#print(np.atleast_2d(x_train), np.atleast_2d(y_train))

def make_2d(arr):
    return [[i] for i in arr]

y_train = make_2d(y_train)
y_test = make_2d(y_test)

model = MLP()
model.fit(np.atleast_2d(x_train), np.atleast_2d(y_train), max_iterations=100, tolerance=1e-4, verbose=True)
print("TRAIN COST: %f" %(model.cost(np.atleast_2d(x_train), np.atleast_2d(y_train))) )
print("TEST COST: %f" %(model.cost(np.atleast_2d(x_test), np.atleast_2d(y_test))))

dump(model, open("./examples/austin_weather_model/austin_weather_model.pickle", "wb"))
