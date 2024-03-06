import json
import queue
import threading
from pytube import YouTube
import time

num_of_threads = 24

def download_videos(video_id, video_info):
    try:
        url = video_info['url']
        file_name = f"{video_id}.mp4"
        
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path='./download_videos', filename=file_name)
    except:
        print(f"Failed to download video: {video_id}")


def worker(q):
    while not q.empty():
        try:
            video_id, video_info = q.get()
            download_videos(video_id, video_info) 
        except:
            print(f"Failed to download video: {video_id}")
        finally:
            q.task_done()

if __name__ == "__main__":
    q = queue.Queue()
    with open('./metafiles/hdvg_0_first_100.json', 'r') as f:
        video_info = json.load(f)

    # with threading
    start_time = time.time()
    for video_id in video_info: 
        q.put((video_id, video_info[video_id]))

    for i in range(num_of_threads):
        threading.Thread(target=worker, daemon=False, args=(q,)).start()

    q.join()
    print(f"Downloaded {len(video_info)} videos in {time.time() - start_time} seconds")


    # without threading
    # start_time = time.time()
    # for video_id in video_info:
    #     try:
    #         download_videos(video_id, video_info[video_id])
    #     except:
    #         print(f"Failed to download video: {video_id}")
    # print(f"Downloaded {len(video_info)} videos in {time.time() - start_time} seconds")
    