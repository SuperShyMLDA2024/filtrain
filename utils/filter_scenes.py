import numpy

# TODO!!!: implement the filtering logic
def filter_scenes(data, n_taken):
    """
    data consist of scene_id keys with dict containing: rgb_diff, mse, cos_sim, avg_velocity
    """

    return data[:n_taken]