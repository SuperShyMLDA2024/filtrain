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
        dg.run_threaded(num_of_clip_splitter_threads=6, 
                        num_of_downloader_threads=2, 
                        num_of_video_splitter_threads=2)

        self.scene_data = []
        for video_id in self.data:
            for clip_id in self.data[video_id]["clip"]:
                for scene in self.data[video_id]["clip"][clip_id]["scene_split"]:
                    scene_dict = {}
                    scene_dict["scene_id"] = scene["clip_id"]
                    scene_dict["video_id"] = video_id
                    scene_dict["clip_id"] = clip_id[:-4]
                    scene_dict["caption"] = scene["caption"]
                    scene_dict["scene_cut"] = scene["scene_cut"]
                    scene_dict["video_path"] = os.path.join('video_clips', video_id, scene_dict["scene_id"] + '.mp4')
                    scene_dict["frames_path"] = os.path.join('frames_output', video_id, scene["clip_id"])

                    if os.path.exists(scene_dict["frames_path"]) and os.path.exists(scene_dict["video_path"]):
                        self.scene_data.append(scene_dict)
    
    def __len__(self):
        return len(self.scene_data)
    
    def __getitem__(self, idx):
        # uncomment to print the scene data
        # print(idx, self.scene_data[idx])

        return self.scene_data[idx]
    
if __name__ == '__main__':
    with open("metafiles/hdvg_0.json", 'r') as f:
        data = json.load(f)
    print("Data loaded")

    dataset = VideoDataset(data, 0, 9)
    print(len(dataset))
    print(dataset[0])

    for data in dataset:
        clip_id = data['scene_id']
        video_id = data['video_id']
        print(video_id, clip_id)
        print("------------------------------------------------------------")