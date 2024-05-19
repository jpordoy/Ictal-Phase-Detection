import pandas as pd
from scipy.interpolate import CubicSpline

class DataFrameInterpolator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def read_csv(self):
        try:
            self.df = pd.read_csv(self.file_path)
            self.df.columns = ['eventID', 'userID', 'date', 'time', 'roi', 'hr', 'output', 'mag', 'outcome']
            start_value = 1  # Start value for the auto-incrementing sequence
            self.df.insert(0, 'Id', range(start_value, start_value + len(self.df)))
            columns_to_keep = ['Id', 'eventID', 'hr', 'mag', 'outcome']
            # Drop columns that are not in the list
            self.df = self.df[columns_to_keep]
            print("CSV file read successfully.")
        except Exception as e:
            print("An error occurred while reading the CSV file:", e)

    def replace_minus_one(self):
        try:
            for index, row in self.df.iterrows():
                if row['hr'] == -1:
                    next_index = index + 1
                    while next_index < len(self.df) and self.df.at[next_index, 'hr'] == -1:
                        next_index += 1
                    if next_index < len(self.df):
                        self.df.at[index, 'hr'] = self.df.at[next_index, 'hr']
                    else:
                        # If no positive number found, leave it as it is
                        pass
            print("Replaced -1 values with next positive number.")
        except Exception as e:
            print("An error occurred while replacing -1 values:", e)

    def interpolate_hr(self):
        try:
            for eventID, group in self.df.groupby('eventID'):
                hr_values = group['hr'].values
                non_glitched_indices = (hr_values != -1).nonzero()[0]
                if len(non_glitched_indices) > 1:
                    spline = CubicSpline(non_glitched_indices, hr_values[non_glitched_indices])
                    interpolated_hr_values = spline(range(len(hr_values)))
                    # Round interpolated values to 3 decimal places
                    interpolated_hr_values = [round(val, 3) for val in interpolated_hr_values]
                    self.df.loc[group.index, 'hr'] = interpolated_hr_values
            print("Interpolated heart rate values.")
        except Exception as e:
            print("An error occurred while interpolating heart rate values:", e)

    def save_to_csv(self, output_file="corrected_dataset.csv"):
        try:
            self.df.to_csv(output_file, index=False, float_format='%.3f')
            print("Dataset corrected successfully and saved as '{}'.".format(output_file))
        except Exception as e:
            print("An error occurred while saving the dataset:", e)

# Example usage:
if __name__ == "__main__":
    data_interpolator = DataFrameInterpolator("ds.csv")
    data_interpolator.read_csv()
    data_interpolator.replace_minus_one()
    data_interpolator.interpolate_hr()
    data_interpolator.save_to_csv()
