from torch.utils.data import Dataset
import json
from get_data_from_idx_range import get_data_from_idx_range
from download_and_cut import download_video

class VideoDataset(Dataset):
    def __init__(self, data, start_idx, end_idx):
        self.list_keys = list(data.keys())
        self.data = get_data_from_idx_range(data, self.list_keys, 
                                             start_idx, end_idx, 
                                             save_to_json=False)
        self.data_list = list(self.data.values())
        download_video(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == '__main__':
    with open("metafiles/hdvg_0.json", 'r') as f:
        data = json.load(f)
    dataset = VideoDataset(data, 0, 99)

    print(len(dataset))
    print(dataset[0])