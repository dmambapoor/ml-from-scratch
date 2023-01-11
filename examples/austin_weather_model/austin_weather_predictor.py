import pandas as pd
import numpy as np
from pickle import load

# Read in the dataset
austin_wdata = pd.read_csv("./examples/austin_weather_model/noaa_austin_2000_present.csv",
                           index_col="DATE", usecols=["DATE", "TMIN", "TMAX", "PRCP"],
                           dtype={"TMIN": np.float64, "TMAX": np.float64, "PRCP": np.float64})

# Note: This data-set does not have null values for TMIN, TMAX.
# run the folllowing commands to verify:
#   print(austin_wdata.apply(pd.isnull).sum())
#   print(austin_wdata.apply(lambda x : (x==9999)).sum()) # '9999' indicates NOAA did not have temperature data for that field.

# Squash data within 0 to 1
squeeze_data = {}
for col in austin_wdata.columns:
    # Get max and min of a column
    min_point = austin_wdata[col].min()
    max_point = austin_wdata[col].max()

    # Store max and min for unsqueezing later
    squeeze_data[col] = (min_point, max_point)

    # Linear squash
    austin_wdata[col] = austin_wdata[col].apply(lambda x: (x - min_point)/(max_point-min_point))

# Add previous [window_size] days of data to each entry
window_size = 6
for col in austin_wdata.columns:
    for i in range(window_size):
        austin_wdata["%s_%i" % (col, i+1)] = austin_wdata.shift((i+1))["TMAX"]

# Add target
austin_wdata["TARGETMAX"] = austin_wdata.shift(-1)["TMAX"]

# Trim rows with not enough previous info to make a weekly window
austin_wdata = austin_wdata.iloc[window_size+1:-1, :].copy()

x_train = austin_wdata.iloc[:int(austin_wdata.shape[0] * 0.8), :-1].to_numpy()
y_train = austin_wdata.iloc[:int(austin_wdata.shape[0] * 0.8), -1].to_numpy()
x_test = austin_wdata.iloc[int(austin_wdata.shape[0] * 0.8):, :-1].to_numpy()
y_test = austin_wdata.iloc[int(austin_wdata.shape[0] * 0.8):, -1].to_numpy()

# Load in trained model
model = load(open("./examples/austin_weather_model/austin_weather_model.pickle", "rb"))

# Predict on training set
y_pred = model.forwardPropagation(x_test)


# Function to return temperatures to original range (rather than [0,1])
def unsqueeze(num, min_point, max_point):
    return num*(max_point-min_point) + min_point


absolute_mean_error = 0.0
for i in range(len(y_pred)):
    y_pred_i = np.round(unsqueeze(y_pred[i], squeeze_data["TMAX"][0], squeeze_data["TMAX"][1]))
    y_test_i = unsqueeze(y_test[i], squeeze_data["TMAX"][0], squeeze_data["TMAX"][1])
    print("Predicted: %f || Actual: %f" % (y_pred_i, y_test_i))
    absolute_mean_error += abs(y_test_i - y_pred_i)

absolute_mean_error /= len(y_pred)
print("ABSOLUTE MEAN ERROR: %f" % (absolute_mean_error))
