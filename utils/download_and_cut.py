import json
import queue
import threading
import yt_dlp
import time
import os
from yt_dlp.utils import download_range_func

temp_dir = "./tmp_clips/"
output_dir = "./video_clips/"
hdvg_dir = './metafiles/hdvg_batch_0-99.json'

num_of_threads = 12

def parse_timestamp(timestamp: str) -> float:
    # timestamp format: HH:MM:SS.MS
    h, m, s = timestamp.split(':')
    s, ms = s.split('.')
    secs = int(h) * 3600 * 1000 + int(m) * 60 * 1000 + int(s) * 1000 + int(ms)
    return secs/1000

class loggerOutputs:
    def debug(msg):
        pass
    def warning(msg):
        pass
    def error(msg):
        pass

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
        'outtmpl': f"{temp_dir}/%(section_start)s {video_id}.mp4",
        'concurrent_fragment_downloads': 3,
        'quiet': True,
        'format': '22',
        'logger': loggerOutputs,
    }
    return opt

def download_clips(video_id, video_info):
    url = video_info['url']
    
    
    with yt_dlp.YoutubeDL(yt_opts(video_id, video_info)) as ydl:
        ydl.download(url)


    files_to_rename = [] # [(old_name, new_name)]

    folder_dir = os.path.join(output_dir, video_id)

    os.makedirs(folder_dir, exist_ok=True)

    for clip_id in video_info["clip"]:
        cur_folder_dir = os.path.join(folder_dir, clip_id)
        start_time = parse_timestamp(video_info["clip"][clip_id]["span"][0])
        old_name = f"{start_time} {video_id}.mp4"      
        new_name = cur_folder_dir
        files_to_rename.append((old_name, new_name))

    return files_to_rename

def worker(q):
    current_video = None
    while not q.empty():
        try:
            video_id, video_info = q.get()
            current_video = video_id
            folder_dir = os.path.join(output_dir, video_id)

            if os.path.exists(folder_dir):
                print(f"Video: {video_id} already downloaded")
                continue

            files_to_rename = download_clips(video_id, video_info)
            
            for files in files_to_rename:
                old_name, new_name = files
                old_name = temp_dir + '/' +  old_name     
                os.rename(old_name, new_name)

        except Exception as e:
            print("Error when downloading video: ", current_video)
            print(e)
        finally:
            q.task_done()

def download_video(data):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    q = queue.Queue()

    for video_id in data:
        q.put((video_id, data[video_id])) # data video_id, clip_id, video_info

    print("Start downloading videos...")
    start_time = time.time()
    for i in range(num_of_threads):
        threading.Thread(target=worker, daemon=False, args=(q,)).start()
    
    q.join()
    end_time = time.time()
    print(f"Downloaded {len(data)} videos in {end_time - start_time} seconds")
    
    return 

if __name__ == "__main__":
    with open(hdvg_dir, 'r') as f:
        data = json.load(f)
    download_video(data)