import json
import numpy as np


class DataLoader2:
    @staticmethod
    def format_data_into_rows(json_data, event_id):
        formatted_rows = []
        for event in json_data:
            if event['id'] == event_id:
                event_id = event['id']
                for datapoint in event['datapoints']:
                    if datapoint['eventId'] == event_id:
                        hr = float(datapoint['hr'])  # Convert HR to float
                        rawData = [float(val) for val in datapoint.get('rawData', [])]  # Convert rawData to float
                        max_acceleration = np.max(rawData)
                        std_deviation = np.std(rawData)
                        #acceleration_between_time_steps = np.diff(rawData)
                        row = (event_id, hr, max_acceleration, std_deviation, rawData)
                        formatted_rows.append(row)
        return formatted_rows

    @staticmethod
    def load_json_data(filename):
        with open(filename) as f:
            return json.load(f)
