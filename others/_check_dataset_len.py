import json
from utils.dataset_class_batch import VideoDataset

if __name__ == '__main__':
    with open('./metafiles/hdvg_0.json', 'r') as f:
        data = json.load(f)

    dataset = VideoDataset(data, 0, 99)
    print(len(dataset))