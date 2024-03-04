import json
import queue
import threading
import yt_dlp
import time
import os
from decimal import *
from yt_dlp.utils import download_range_func
from yt_dlp import YoutubeDL
from pytube import YouTube

temp_dir = "D:/tmp_clips"
output_dir = "D:/video_clips"
num_of_threads = 24
downloading_done = False

def parse_timestamp(timestamp: str) -> float:
    # timestamp format: HH:MM:SS.MS
    h, m, s = timestamp.split(':')
    s, ms = s.split('.')
    secs = int(h) * 3600 * 1000 + int(m) * 60 * 1000 + int(s) * 1000 + int(ms)
    return secs/1000

def yt_opts(video_id, video_info):
    
    ranges = []

    for clip_id in video_info["clip"]:
        start = parse_timestamp(video_info["clip"][clip_id]["span"][0])
        end = parse_timestamp(video_info["clip"][clip_id]["span"][1])

        ranges.append((start, end))
    # download video-only in mp4
    opt = {
        'verbose': True,
        'download_ranges': download_range_func(None, ranges),
        # 'force_keyframes_at_cuts': True,
        'quiet': True,
        'outtmpl': f"{temp_dir}/%(section_start)s {video_id}.mp4",
        'format': "22/17/18",
        'concurrent_fragment_downloads': 3,
    }
    return opt

def download_clips(video_id, video_info):
    url = video_info['url']
    
    with yt_dlp.YoutubeDL(yt_opts(video_id, video_info)) as ydl:
        ydl.download(url)

    print(f"Downloaded video: {video_id}")

    files_to_rename = [] # [(old_name, new_name)]

    for clip_id in video_info["clip"]:
        start_time = parse_timestamp(video_info["clip"][clip_id]["span"][0])
        old_name = f"{start_time} {video_id}.mp4"      
        new_name = f"{clip_id}"
        files_to_rename.append((old_name, new_name))

    return files_to_rename

def worker(q, q_rename):
    while not q.empty():
        try:
            video_id, video_info = q.get()
            files_to_rename = download_clips(video_id, video_info)
            

            for files in files_to_rename:
                q_rename.put(files)
            
            print("Downloaded video: " + video_id )

        except Exception as e:
            print(e)
        finally:
            q.task_done()

def rename_worker(q):
    print("Start renaming videos...")
    while not downloading_done or not q.empty():
        try:
            old_name, new_name = q.get()
            print("[Test]", old_name, new_name)
            old_name = temp_dir + '/' +  old_name     
            new_name = output_dir + '/' +  new_name
            print(f"Renaming {old_name} to {new_name}")
            os.rename(old_name, new_name)
        except Exception as e:
            print(e)
        finally:
            q.task_done()

if __name__ == "__main__":

    os.rmdir(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    getcontext().prec = 3
    q = queue.Queue()
    q_rename = queue.Queue()

    with open('./metafiles/hdvg_0_first_100.json', 'r') as f:
        video_info = json.load(f)
    for video_id in video_info:
        q.put((video_id, video_info[video_id])) # data video_id, clip_id, video_info

    print("Start downloading videos...")
    start_time = time.time()
    for i in range(num_of_threads):
        threading.Thread(target=worker, daemon=False, args=(q,q_rename,)).start()
    
    for i in range(3):
        threading.Thread(target=rename_worker, daemon=False, args=(q_rename,)).start()
    
    q.join()
    downloading_done = True
    q_rename.join()
    end_time = time.time()
    print(f"Downloaded {len(video_info)} videos in {end_time - start_time} seconds")
    
    