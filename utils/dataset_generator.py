import json
import queue
from pytube import YouTube
import cv2
import subprocess
import time
import os
import shutil
import multiprocessing as mp



class DatasetGenerator:
    def download_worker(self, q, vq, initialize_finished):
        while not q.empty() or not initialize_finished.is_set():
            try:
                video_id, video_info = q.get()
                self.download_video(video_id, video_info)

                for clip_id, clip_info in video_info['clip'].items():
                    vq.put((video_id, clip_id, clip_info))
                print("Done downloading video: " + video_id)
            except Exception as e:
                print(f"Error when downloading video: {video_id}")
            finally:
                q.task_done()
        
        return

    def video_split_worker(self,q, sq, download_finished):
        while not q.empty() or not download_finished.is_set():
            try:
                video_id, clip_id, clip_info = q.get()
                self.split_video(clip_info, video_id, clip_id)
                for scene in clip_info['scene_split']:
                    sq.put((video_id, clip_id, scene))
                print("Done splitting video: " + video_id)
            except Exception as e:
                print(f"Error when splitting video: {video_id}")
            finally:
                q.task_done()
        
        return


    def clip_split_worker(self,q, video_split_finished):
        while not q.empty() or not video_split_finished.is_set():
            try:
                video_id, clip_id, info_scene = q.get()
                self.split_clip(info_scene, self.data[video_id]['clip'][clip_id]['fps'], video_id, clip_id, info_scene["clip_id"])

                print("Done splitting clip: " + video_id + "/" + clip_id + "/" + info_scene["clip_id"])
            except Exception as e:
                print(f"Error when splitting clip: {video_id}/{clip_id}")
            finally:
                q.task_done()
        
        return

    def __init__(self, filename = None, generate_scene_video = True, generate_scene_samples = False, frame_output_folder = "frames_output", scenes_output_folder = "video_clips", download_output_folder="download_videos", tmp_output_folder="tmp_clips"):
        self.filename = filename
        self.generate_scene_video = generate_scene_video
        self.generate_scene_samples = generate_scene_samples
        self.frame_output_folder = frame_output_folder
        self.scenes_output_folder = scenes_output_folder
        self.download_output_folder = download_output_folder
        self.tmp_output_folder = tmp_output_folder
        self.data = None

        os.makedirs(download_output_folder, exist_ok=True)
        os.makedirs(tmp_output_folder, exist_ok=True)

        if self.generate_scene_video:
            os.makedirs(scenes_output_folder, exist_ok=True)

        if self.generate_scene_samples:
            os.makedirs(self.frame_output_folder, exist_ok=True)
        

    def cmd(self, cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        return out.decode('utf-8')

    def hhmmss(self, timestamp1, timestamp2):
        hh,mm,s = timestamp1.split(':')
        ss,ms = s.split('.')
        timems1 = 3600*1000*int((hh)) +  60*1000*int(mm) + 1000*int(ss) + int(ms)
        hh,mm,s = timestamp2.split(':')
        ss,ms = s.split('.')
        timems2 = 3600*1000*int((hh)) +  60*1000*int(mm) + 1000*int(ss) + int(ms)
        dur = (timems2 - timems1)/1000
        return str(dur)

    def download_video(self, video_id, video_info):
        # Checks if the video is already downloaded
        if os.path.exists(os.path.join(self.download_output_folder, f"{video_id}.mp4")):
            return
        
        url = video_info['url']
        file_name = f"{video_id}.mp4"
        
        yt = YouTube(url)
        stream = yt.streams.get_by_resolution('360p')
        stream.download(output_path=self.download_output_folder, filename=file_name)

    def split_video(self, info, video_id, output_name):
        # info: a unit of a clip (span, scene_split, fps)
        # video_name = the name of the downloaded video
        # output_name = the name of the outputted video
        # cut hdvila clip
        yt_video = os.path.join(self.download_output_folder, video_id +'.mp4')

        ori_clip_path = os.path.join(self.tmp_output_folder, output_name) # output the clip videos on temporary folder
        os.makedirs(os.path.join(self.tmp_output_folder, video_id), exist_ok=True)
        if not os.path.exists(ori_clip_path):
            sb = info['span']
            cmd = ['ffmpeg', '-ss', sb[0], '-t', self.hhmmss(sb[0], sb[1]),'-accurate_seek', '-i', yt_video, '-c', 'copy',
                '-avoid_negative_ts', '1', '-reset_timestamps', '1',
                '-y', '-hide_banner', '-loglevel', 'panic', '-map', '0', ori_clip_path]
            self.cmd(cmd)

        if not os.path.isfile(ori_clip_path):
            raise Exception(f"{ori_clip_path}: ffmpeg clip extraction failed")

    def split_clip(self, info_scene, fps, video_id, clip_input_name, scene_output_name):
        # info: a unit of a scene
        ori_clip_path = os.path.join(self.tmp_output_folder, clip_input_name)

        clip_id = clip_input_name[:-4]
        
        try:
            start, end = int(info_scene['scene_cut'][0]), int(info_scene['scene_cut'][1])
            save_split_path = os.path.join(self.scenes_output_folder, video_id, scene_output_name + '.mp4')

            # Skip if the scene is already split
            if os.path.exists(save_split_path):
                return

            os.makedirs(os.path.join(self.scenes_output_folder, video_id), exist_ok=True)
            if end == -1:
                shutil.copy(ori_clip_path, save_split_path)
            else:
                oricap = cv2.VideoCapture(ori_clip_path)
                h = oricap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                w = oricap.get(cv2.CAP_PROP_FRAME_WIDTH)

                writer = cv2.VideoWriter(save_split_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(w),int(h)))
                oricap.set(cv2.CAP_PROP_POS_FRAMES, start+1)
                current = start+1
                frame_cnt = 0
                
                if self.generate_scene_samples:
                    os.makedirs(os.path.join(self.frame_output_folder, video_id, scene_output_name), exist_ok=True)

                while current < end:
                    ret, frame = oricap.read()

                    if self.generate_scene_samples and (frame_cnt * 500 < current / fps * 1000):
                        frame_name = os.path.join(self.frame_output_folder, video_id, scene_output_name, f"{frame_cnt:04d}.jpg")
                        cv2.imwrite(frame_name, frame)
                        frame_cnt += 1
                    
                    if self.generate_scene_video and ret:
                        writer.write(frame)
                    current += 1
                writer.release()
                oricap.release()

        except Exception as e:
            print("Error occured")
            print(e)

    def load_data(self, data):
        self.data = data
        # print("Metadata loaded (data)")

    def load_from_file(self):
        with open(self.filename, 'r') as f:
            self.data = json.load(f)
        # print("Metadata loaded (from files)")
    
    def download(self):
        for video_id, video_info in self.data.items():
            self.download_video(video_id, video_info)
        # print("Done downloading")

    def split_videos(self):
        for video_id, video_info in self.data.items():
            for clip_id, clip_info in video_info['clip'].items():
                self.split_video(clip_info, video_id, clip_id)
            # print("Done splitting videos: " + video_id)

            # Can delete video
        # print("Done splitting videos")
    
    def split_scenes(self):
        for video_id, video_info in self.data.items():
            for clip_id, clip_info in video_info['clip'].items():
                for scene in clip_info['scene_split']:
                    self.split_clip(scene, clip_info['fps'], video_id, clip_id, scene["clip_id"])
                    # print("Done splitting clips: " + scene["clip_id"])
        # print("Done splitting clips")

                
    
    def run(self):
        self.load_from_file()
        self.download()
        self.split_videos()
        self.split_scenes()
        print("Done")

        
    def run_threaded(self, num_of_downloader_threads = 2, num_of_video_splitter_threads = 2, num_of_clip_splitter_threads = 4):

        if self.filename is not None:
            self.load_from_file()

        dq = mp.JoinableQueue() # video id, video info
        vq = mp.JoinableQueue()
        sq = mp.JoinableQueue()

        initialize_finished = mp.Event()
        download_finished = mp.Event()
        video_split_finished = mp.Event()

        processes = []

        for _ in range(num_of_downloader_threads):
            t = mp.Process(target=self.download_worker, args=(dq, vq, initialize_finished))
            t.start()
            processes.append(t)     
        print("Spawned download workers")

        for _ in range(num_of_video_splitter_threads):
            t = mp.Process(target=self.video_split_worker, args=(vq, sq, download_finished,))
            t.start()
            processes.append(t)
        print("Spawned video split workers")

        for _ in range(num_of_clip_splitter_threads):
            t = mp.Process(target=self.clip_split_worker, args=(sq, video_split_finished,))
            t.start()
            processes.append(t)
        print("Spawned clip split workers")

        for video_id, video_info in self.data.items():
            dq.put((video_id, video_info))
        
        initialize_finished.set()
        print("Initialization finished")        
        dq.join()
        print("Download finished")
        download_finished.set()
        vq.join()
        print("Video split finished")
        video_split_finished.set()
        sq.join()
        print("Clip split finished")

        for p in processes:
            if p.is_alive():
                p.terminate()

        return


if __name__ == '__main__':
    dg = DatasetGenerator("metafiles/hdvg_batch_0-1.json", generate_scene_samples=True, generate_scene_video=True)
    dg.run_threaded()