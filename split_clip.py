import ffmpeg
import json
import os
import queue
import threading

hdvg_dir = "./metafiles/hdvg_batch_0-99.json"
clip_dir = "./video_clips"
out_dir = "./frames_output"
time_interval_ms = 250

num_of_threads = 12
finished = False

def split_clip(clip_id, scenes_details):
    # scene_split = [
    #     scene_cut: [start, end],
    #     clip_id: clip_id, // output video
    #     caption: caption
    # ]

    # cut hdvg clip
    fps = scenes_details['fps']
    scene_splits = scenes_details['scene_split']
    for scene in scene_splits:
        
        start = scene['scene_cut'][0]
        end = scene['scene_cut'][1]
        caption = scene['caption']
        scene_id = scene['clip_id']
        os.makedirs(f"{out_dir}/{scene_id}", exist_ok=True)

        output_dir = f'{out_dir}/{scene_id}/%04d.png'
        input_dir = f"{clip_dir}/{clip_id}"

        ffmpeg.input(input_dir)\
            .trim(start_frame=start, end_frame=end)\
            .filter('fps', fps=1/(time_interval_ms/1000), round='up')\
            .setpts('PTS-STARTPTS')\
            .output(output_dir)\
            .overwrite_output()\
            .run()
        
        print(f'clip_id: {clip_id} start: {start} end: {end} caption: {caption}')

    return

def worker(q):
    while not q.empty() or not finished:
        try:
            clip_id, scenes_details = q.get()
            split_clip(clip_id, scenes_details)
            print(f"Done clip: {clip_id}")
        except Exception as e:
            print(e)
        finally:
            q.task_done()

if __name__ == "__main__":
    with open(hdvg_dir, 'r') as f:
        video_info = json.load(f)

    q = queue.Queue()

    for _ in range(num_of_threads):
        threading.Thread(target=worker, daemon=False, args=(q,)).start()
    
    for video_id in video_info:
        clips = video_info[video_id]['clip']
        for clip_id in clips:
            q.put((clip_id, clips[clip_id]))
            # split_clip(clip_id, clips[clip_id])
            break
    
    q.join()
    
    print("Done")
