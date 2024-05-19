# Function to format data into individual rows for event 5580
# Function to format data into individual rows for event 5580
def format_data_into_rows(json_data):
    formatted_rows = []
    for event in json_data:
        if event['id'] == 12763:
            event_id = event['id']
            for datapoint in event['datapoints']:
                if datapoint['eventId'] == 12763:
                    eventId = datapoint['eventId']
                    hr = datapoint['hr']
                    rawData = datapoint['rawData']
                    formatted_rows.append((hr, rawData)) 
    return formatted_rows


