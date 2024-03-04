import json
from pytube import YouTube
def download_videos(json_file):
    with open(json_file) as file:
        data = json.load(file)
        
        for video_id, video_info in data.items():
            url = video_info['url']
            file_name = f"{video_id}.mp4"
            
            yt = YouTube(url)
            yt.streams.get_highest_resolution().download(output_path='./download_videos', filename=file_name)
            print(f"Downloaded video: {file_name}")

# Usage
json_file = './metafiles/hdvg_0.json'
download_videos(json_file)