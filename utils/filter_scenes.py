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
    data_ids = []
    clip_ids = []
    for scene_data in data:
        data_ids.append(scene_data['idx'])
        clip_ids.append(scene_data['clip_id'])
        scene_data.pop('idx')
        scene_data.pop('clip_id')

    # convert to dataframe
    data = pd.DataFrame(data)

    # get the prediction
    pred = classifier_model.predict_proba(data)[:, 1]

    # sort data_ids and clip_ids based on pred descending
    sorted_data_ids = numpy.argsort(pred)[::-1]
    sorted_clip_ids = [clip_ids[i] for i in sorted_data_ids]
    sorted_data_ids = [data_ids[i] for i in sorted_data_ids]

    # take the top n_taken but skip clips that have been taken
    filtered_data = []
    clips_taken = []
    for i in range(len(data)):
        if len(filtered_data) == n_taken:
            break
        if sorted_clip_ids[i] not in clips_taken:
            filtered_data.append(sorted_data_ids[i])
            clips_taken.append(sorted_clip_ids[i])

    
    return filtered_data


# Example usage
if __name__ == '__main__':
    # load inference_result_0-0.json
    with open('frame_score_results/inference_result_100-100.json', 'r') as f:
        data = json.load(f)

    # load the classifier model
    classifier_model = pickle.load(open('xgboost_model.pkl', 'rb'))

    # filter the scenes
    filtered_scenes = filter_scenes(data, 2, classifier_model)
    print(filtered_scenes)
