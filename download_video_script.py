import json
from pytube import YouTube
def download_videos(video_id, video_info):
    url = video_info['url']
    file_name = f"{video_id}.mp4"
    
    yt = YouTube(url)
    yt.streams.get_highest_resolution().download(output_path='./download_videos', filename=file_name)
    print(f"Downloaded video: {file_name}")

if __name__ == "__main__":
    with open('./metafiles/hdvg_0_first_10.json', 'r') as f:
        video_info = json.load(f)
    for video_id in video_info:
        download_videos(video_id, video_info[video_id])