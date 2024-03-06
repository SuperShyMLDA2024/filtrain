import numpy
import pandas as pd
import json
import pickle

def filter_scenes(data, n_taken, classifier_model):
    """
    data consist of scene_id keys with dict containing: rgb_diff, avg_velocity, mse, cos_sim
    """

    # data is dict of dict and we want to convert it to list of dict
    data = list(data.values())
    data_info = []
    for keys in data:
        data_info.append(keys['info'])
        keys.pop('info')

    # convert to dataframe
    data = pd.DataFrame(data)

    # get the prediction
    pred = classifier_model.predict_proba(data)[:, 1]

    # sort the data_info based on the prediction values
    data_info = [x for _, x in sorted(zip(pred, data_info), key=lambda pair: pair[0], reverse=True)]

    # return the top n_taken scene_ids
    return data_info[:n_taken]


# Example usage
if __name__ == '__main__':
    # load inference_result_0-0.json
    with open('frame_score_results/inference_result_100-100.json', 'r') as f:
        data = json.load(f)

    # load the classifier model
    classifier_model = pickle.load(open('xgboost_model.pth', 'rb'))

    # filter the scenes
    filtered_scenes = filter_scenes(data, 2, classifier_model)
    print(filtered_scenes)
