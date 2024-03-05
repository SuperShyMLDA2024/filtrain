from torch.utils.data import Dataset
import os
import json
from get_data_idx_range import get_data_idx_range
from download_and_cut import download_video
from split_clip import split_clip

class VideoDataset(Dataset):
    def __init__(self, data, start_idx, end_idx):
        self.list_keys = list(data.keys())
        self.data = get_data_idx_range(data, self.list_keys, 
                                       start_idx, end_idx, 
                                       save_to_json=False)
        self.data_list = list(self.data.values())
        download_video(self.data)
        self.clip_data = []
        for video_id in self.data:
            for clip_id in self.data[video_id]["clip"]:
                clip_dict = self.data[video_id]["clip"][clip_id]
                clip_dict["clip_name"] = clip_id
                clip_dict["file_path"] = os.path.join('video_clips', video_id, clip_id) 
                clip_dict["video_id"] = video_id
                if os.path.exists(clip_dict["file_path"]):
                    self.clip_data.append(clip_dict)
    
    def __len__(self):
        return len(self.clip_data)
    
    def __getitem__(self, idx):
        video_id = self.clip_data[idx]["video_id"]
        clip_id = self.clip_data[idx]["clip_name"]
        scenes_details = self.clip_data[idx]
        split_clip(video_id, clip_id, scenes_details)
        return self.clip_data[idx]
    
if __name__ == '__main__':
    with open("metafiles/hdvg_0.json", 'r') as f:
        data = json.load(f)
    print("Data loaded")

    dataset = VideoDataset(data, 0, 9)
    print(len(dataset))
    print(dataset[0])