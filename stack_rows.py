import pandas as pd
import json

def format_data_into_rows(json_data):
    formatted_rows = []
    for event in json_data:
        if event['id'] == 42147:
            event_id = event['id']
            for datapoint in event['datapoints']:
                if datapoint['eventId'] == 42147:
                    hr = datapoint['hr']
                    rawData = datapoint['rawData']
                    for raw_value in rawData:
                        formatted_rows.append((hr, raw_value))  # Append as tuple
    return formatted_rows

def main():
    # Load JSON data from file (replace 'dataset.json' with your actual file path)
    with open('dataset.json') as f:
        json_data = json.load(f)

    # Format data into individual rows
    formatted_data = format_data_into_rows(json_data)

    # Create DataFrame
    df = pd.DataFrame(formatted_data, columns=['Hr', 'RawData'])

    # Split the 'RawData' column and stack it vertically downwards
    raw_data_stacked = df['RawData'].explode()

    # Create a DataFrame for the stacked 'RawData' column
    raw_data_df = pd.DataFrame({'RawData': raw_data_stacked})

    # Concatenate the 'HR' column with the stacked 'RawData' column vertically
    stacked_df = pd.concat([df['Hr'], raw_data_df], axis=1)

    # Save the vertically stacked DataFrame to a new CSV file
    output_file_path = 'New.csv'
    stacked_df.to_csv(output_file_path, index=False)

if __name__ == '__main__':
    main()
