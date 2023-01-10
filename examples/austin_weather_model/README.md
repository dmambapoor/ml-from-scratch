# Austin Weather Model

## Introduction
This model takes NOAA data from Austin Bergstrom Airport. It uses data from some adjustable number of previous days. The measurements is as follows:
* daily temperature high
* daily temperature low
* daily precipitation

This model uses the [MLP](src/ml_from_scratch/MLP.py) model from the src folder.

## How to run the model
1. Clone the repo or download the examples folder.
1. Install the ml_from_scratch package.
    * If you've cloned the repo simply use `pip install` with the root as your current directory.
    * If you've only downloaded the example folder, then use `pip install git+https://github.com/dmambapoor/ml-from-scratch.git`
1. Ensure your directory structure looks like the following:
```
parent-folder/
    examples/
        austin_weather_model/
            austin_weather_model_trainer.py
            austin_weather_model.pickle
            austin_weather_predictor.py
            noaa_austin_2000_present.csv
        ...
    ...
```
4. Install a python 3.10+ version.
1. Run the following python command from the parent/root directory: 
```bash
# Train the model
python3 ./examples/austin_weather_model/austin_weather_model_trainer.py

# Run the model
python3 ./examples/austin_weather_model/austin_weather_model_predictor.py
```

## Lessons Learned
* **Refer to data source documentation** to find null entries. Some null entries may be filled with a special code.
* **The initial state of the model greatly affects the outcome of gradient descent.** If a solution space is filled with many local minima then the starting space effectively selects for those minima by proximity.
* **Learning rate adjustment doesn't significantly affect the outcome of gradient descent** (assuming the adjustments only reduce the rate).