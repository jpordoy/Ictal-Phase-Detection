import pandas as pd

# Load the original CSV file
df = pd.read_csv('ds1.csv')

# Initialize lists to store transposed data
transposed_hr = []
transposed_rawData = []
transposed_eventID = []
transposed_label = []


# Iterate over each row of the DataFrame
for index, row in df.iterrows():
    # Repeat eventId and label values 125 times
    repeated_eventID = [row['eventID']] * 125
    repeated_label = [row['label']] * 125

    # Transpose hr and rawData values
    transposed_hr.extend([row['hr']] * 125)
    rawData_values = row['rawData'].strip('][').split(', ')
    transposed_rawData.extend(rawData_values)

    # Extend the lists for eventID and label
    transposed_eventID.extend(repeated_eventID)
    transposed_label.extend(repeated_label)

# Create a new DataFrame with transposed data
new_df = pd.DataFrame({'eventID': transposed_eventID, 'hr': transposed_hr, 'rawData': transposed_rawData, 'label': transposed_label})

# Save the new DataFrame to a new CSV file
new_df.to_csv('transposed_data_with_hr.csv', index=False)
