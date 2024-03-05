# read mlda_data.json
# get first n records
# write to mlda_data_first_n.json

import json

def get_first_n_records(n, json_filename='hdvg_0'):
    # Read the data from the JSON file
    with open("metafiles/" + json_filename + '.json', 'r') as f:
        data = json.load(f)

    first_n_records = {}
    for video in data:
        first_n_records[video] = data[video]
        if len(first_n_records) == n:
            break

    # Write the first n records to a new JSON file
    with open(f'metafiles/{json_filename}_first_{n}.json', 'w') as f:
        json.dump(first_n_records, f)

# Call the function with the number of records you want
get_first_n_records(100)