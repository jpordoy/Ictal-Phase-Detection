import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import pandas as pd
import csv
import numpy as np
import pandas as pd
from scipy import stats


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
import re


global_variable = 5254  # Define a global variable outside any function
csv_file_path = f"Training/{global_variable}.csv"


def format_data_into_arrays(json_data):
    hr_array = []
    rawData_array = []
    for event in json_data:
        if event['id'] == global_variable:
            event_id = event['id']
            for datapoint in event['datapoints']:
                if datapoint['eventId'] == global_variable:
                    hr = datapoint['hr']
                    rawData = datapoint['rawData']
                    hr_array.append(hr)
                    rawData_array.append(rawData)
    return hr_array, rawData_array


def correct_glitched_hr(array):
    corrected_values = []
    for i, hr in enumerate(array):
        if hr == -1:
            # Find the next non-glitched heart rate value within the same timestep
            next_valid_hr_index = i + 1
            next_valid_hr = next((hr for hr in array[next_valid_hr_index:] if hr != -1), None)
            if next_valid_hr is not None:
                # Generate artificial data based on the next valid heart rate value
                corrected_values.append(next_valid_hr)
            else:
                # If there are no more valid heart rate values, set it to 0
                corrected_values.append(0)  # Or any default value as needed
        else:
            corrected_values.append(hr)
    return corrected_values


def replace_trailing_zeros(array):
    # Find the index of the last positive integer value
    last_positive_index = len(array) - 1
    while last_positive_index >= 0 and array[last_positive_index] == 0:
        last_positive_index -= 1

    # Replace trailing zeros with decreasing positive integer values
    current_value = array[last_positive_index]
    for i in range(last_positive_index + 1, len(array)):
        array[i] = current_value
        current_value -= 1
        if current_value < 0:
            current_value = 0  # Ensure non-negative values
    
    return array




def main():
    # Load JSON data
    with open('Data/OSDB.v3_Dataset_Full/dataset.json') as f:
        json_data = json.load(f)


    # Call format_data_into_arrays function
    hr_array, rawData_array = format_data_into_arrays(json_data)
    hr_array = np.array(hr_array)
    replaced_array = correct_glitched_hr(hr_array)
    hr_array = replace_trailing_zeros(replaced_array)
    
    
    print(hr_array)
    
    # Repeat each value 125 times
    repeated_hr_array = np.repeat(hr_array, 125)
    print(repeated_hr_array)
    print(len(hr_array)) 
    print(len(repeated_hr_array))

   
if __name__ == '__main__':
    main()