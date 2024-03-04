import json
import queue
import threading
from pytube import YouTube


num_of_threads = 16

def download_videos(video_id, video_info):
    url = video_info['url']
    file_name = f"{video_id}.mp4"
    
    yt = YouTube(url)
    yt.streams.get_highest_resolution().download(output_path='./download_videos', filename=file_name)
    print(f"Downloaded video: {file_name}")


def worker(q):
    while True:
        video_id, video_info = q.get()
        download_videos(video_id, video_info)
        q.task_done()

if __name__ == "__main__":
    q = queue.Queue()
    with open('./metafiles/hdvg_0.json', 'r') as f:
        video_info = json.load(f)
    for video_id in video_info:
        q.put((video_id, video_info[video_id]))

    for i in range(num_of_threads):
        threading.Thread(target=worker, daemon=False, args=(q,)).start()
    
    q.join()
    
    