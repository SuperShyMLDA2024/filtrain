# read mlda_data.json
# get random n records
# write to mlda_data_first_n.json
import json
import random 

def get_random_n_records(n, json_filename='hdvg_0', seed=0):
    # Set the seed for reproducibility
    random.seed(seed)
    
    # Read the data from the JSON file
    with open("metafiles/" + json_filename + '.json', 'r') as f:
        data = json.load(f)

    list_keys = list(data.keys())
    # take random n records from list_keys
    random_n_keys = random.sample(list_keys, n)
    random_n_records = {}
    for key in random_n_keys:
        random_n_records[key] = data[key]

    # Write the random n records to a new JSON file
    with open(f'metafiles/{json_filename}_random_{n}.json', 'w') as f:
        json.dump(random_n_records, f)

# Call the function with the number of records you want
get_random_n_records(10)