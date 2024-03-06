import numpy

# TODO!!!: implement the filtering logic
def filter_scenes(data, n_taken):
    """
    data consist of scene_id keys with dict containing: rgb_diff, avg_velocity, mse, cos_sim
    """

    filtered_data = []
    for scene_id in data:
        filtered_data.append(scene_id)

    return filtered_data[:n_taken]