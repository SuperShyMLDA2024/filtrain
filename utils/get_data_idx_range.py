# read mlda_data.json
# get videos per batch

import json

def get_data_idx_range(data, list_keys, start_idx, end_idx, to_list=True, save_to_json=False):
    keys = list_keys[start_idx:end_idx+1]

    # take random n records from list_keys
    records = {}
    for key in keys:
        records[key] = data[key]

    if save_to_json:
        # Write the random n records to a new JSON file
        with open(f'metafiles/hdvg_batch_{start_idx}-{end_idx}.json', 'w') as f:
            json.dump(records, f)
    
    return records

if __name__ == "__main__":
    # Read the data from the JSON file
    with open("metafiles/hdvg_0.json", 'r') as f:
        data = json.load(f)

    list_keys = list(data.keys())
    # Call the function with the number of records you want
    get_data_from_idx_range(data, list_keys, 0, 99, to_list=False, save_to_json=True)