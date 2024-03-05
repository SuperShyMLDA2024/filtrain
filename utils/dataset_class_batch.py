from torch.utils.data import Dataset
from dataset_generator import DatasetGenerator
import os
import json
from get_data_idx_range import get_data_idx_range
from download_and_cut import download_video
from split_clip import split_clip

class VideoDataset(Dataset):
    def __init__(self, data, start_idx, end_idx):
        dg = DatasetGenerator(generate_scene_samples=True, generate_scene_video=True)
        self.list_keys = list(data.keys())
        self.data = get_data_idx_range(data, self.list_keys, 
                                       start_idx, end_idx, 
                                       save_to_json=False)
        self.data_list = list(self.data.values())
        dg.load_data(self.data)
        dg.download()
        dg.split_videos()
        dg.split_scenes()

        self.scene_data = []
        for video_id in self.data:
            for clip_id in self.data[video_id]["clip"]:
                for scene in self.data[video_id]["clip"][clip_id]["scene_split"]:
                    scene_dict = self.data[video_id]["clip"][clip_id]["scene_split"][scene]
                    scene_dict["video_path"] = os.path.join('video_clips', video_id, clip_id + '.mp4')
                    scene_dict["samples_path"] = os.path.join('frames_output', video_id, clip_id, scene)
                    if os.path.exists(scene_dict["samples_path"]) and os.path.exists(scene_dict["video_path"]):
                        self.scene_data.append(scene_dict)
    
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