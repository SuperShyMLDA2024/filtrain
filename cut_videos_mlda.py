import jsonlines
import os
from tqdm import tqdm
import logging
import argparse
import re
import subprocess
import multiprocessing
import os
import json
import shutil
import cv2
from download_video_script import download_videos

def parse_args():
    parser = argparse.ArgumentParser(description='youtube video processing')
    parser.add_argument('--workdir', default='./',type=str, help='Working Directory')
    parser.add_argument('--metafile', default='hdvg_0_first_10.json', type=str, help='youtube video meta')
    parser.add_argument('--resultfile', default='cut_part0.jsonl', type=str, help='processed videos')
    parser.add_argument('--log', default='log_part0.log', type=str, help='log')
    parser.add_argument('--rm_tmp_file', default=True, type=bool, help='Whether to remove tmp hdvila clips')
    args = parser.parse_args()
    return args


def check_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)


class Cutvideos():
    def __init__(self, metafile, workdir, resultfile, rm_tmp_file):
        self.workdir = workdir
        self.metafile = metafile
        self.resultfile = resultfile
        self.rm_tmp_file = rm_tmp_file
        self.metas = self.loadmetas()

    def loadmetas(self):
        print(self.metafile)
        with open(self.metafile, 'r') as f:
            metas = json.load(f)
        return metas

    def hhmmss(self, timestamp1, timestamp2):
        hh,mm,s = timestamp1.split(':')
        ss,ms = s.split('.')
        timems1 = 3600*1000*int((hh)) +  60*1000*int(mm) + 1000*int(ss) + int(ms)
        hh,mm,s = timestamp2.split(':')
        ss,ms = s.split('.')
        timems2 = 3600*1000*int((hh)) +  60*1000*int(mm) + 1000*int(ss) + int(ms)
        dur = (timems2 - timems1)/1000
        return str(dur)

    def run(self, cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        return out.decode('utf-8')

    def extract_single_ori_clip(self, yt_video_id, yt_video, out_folder, tmp_folder, ori_clip, info):
        result = []

        # cut hdvila clip
        ori_clip_path = os.path.join(tmp_folder, ori_clip)
        if not os.path.exists(ori_clip_path):
            sb = info['span']
            cmd = ['ffmpeg', '-ss', sb[0], '-t', self.hhmmss(sb[0], sb[1]),'-accurate_seek', '-i', yt_video, '-c', 'copy',
                '-avoid_negative_ts', '1', '-reset_timestamps', '1',
                '-y', '-hide_banner', '-loglevel', 'panic', '-map', '0',ori_clip_path]
            self.run(cmd)
        if not os.path.isfile(ori_clip_path):
            raise Exception(f"{ori_clip_path}: ffmpeg clip extraction failed")
            logger.info(f"{ori_clip_path}: ffmpeg clip extraction failed")
            return result
        
        # cut hdvg clip
        fps = info['fps']
        scene_splits = info['scene_split']
        for split in scene_splits:
            try:
                caption = split['caption']
                start, end = int(split['scene_cut'][0]), int(split['scene_cut'][1])
                save_split_path = os.path.join(out_folder, split['clip_id'] + '.mp4')
                if end == -1:
                    shutil.copy(ori_clip_path, save_split_path)
                else:
                    oricap = cv2.VideoCapture(ori_clip_path)
                    h = oricap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    w = oricap.get(cv2.CAP_PROP_FRAME_WIDTH)

                    writer = cv2.VideoWriter(save_split_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(w),int(h)))
                    oricap.set(cv2.CAP_PROP_POS_FRAMES, start+1)
                    current = start+1
                    while current < end:
                        ret, frame = oricap.read()
                        if ret:
                            writer.write(frame)
                        current += 1
                result.append({'clip': yt_video_id+'/'+split['clip_id']+'.mp4', 'caption':caption})
            except Exception as e:
                logger.info(e)
        return result

    def extract_clips(self, video_id, meta):
        # extract_clips from a single youtube video
        clips = meta['clip']
        outfolder = os.path.join(self.workdir,'video_clips', video_id)
        tmp_folder = os.path.join(self.workdir,'tmp_clips', video_id)
        check_dirs(outfolder)
        check_dirs(tmp_folder)
        result = []
        # try:
        for ori_clip, info in clips.items():
            re = self.extract_single_ori_clip(video_id, os.path.join(self.workdir,'download_videos', video_id + '.mp4'), outfolder, tmp_folder, ori_clip, info)
            result.extend(re)
        # except:
        #     pass
        if self.rm_tmp_file:
            shutil.rmtree(tmp_folder)
        return result

    def extract_all_clip(self):
        print("Start extracting clips...")
        results = []
        for video_id, meta in tqdm(self.metas.items()):
            download_videos(video_id, meta)
            print(os.path.join(self.workdir,'download_videos', video_id + '.mp4'))
            if not os.path.exists(os.path.join(self.workdir,'download_videos', video_id + '.mp4')):
                logger.info(f'Video missing: {video_id}')
            else:
                result = self.extract_clips(video_id, meta)
                results.extend(result)

        logger.info(f"Number of clips processed: {len(results)}")
        if self.rm_tmp_file:
            shutil.rmtree(os.path.join(self.workdir,'tmp_clips'))

        with jsonlines.open(os.path.join(self.workdir, 'hdvg_results', self.resultfile), 'w') as f:
            for l in results:
                f.write(l)
        

if __name__ == '__main__':
    args = parse_args()
    
    metafile = os.path.join(args.workdir, 'metafiles', args.metafile)
    logdir = os.path.join(args.workdir,'cut_video_log')
    redir = os.path.join(args.workdir, 'hdvg_results')

    check_dirs(os.path.join(args.workdir, 'video_clips'))   # hdvg_root
    check_dirs(os.path.join(args.workdir, 'cut_video_results'))  # log_processed_video, also the video-text info for training
    check_dirs(os.path.join(args.workdir, 'tmp_clips'))   #hdvila_root
    check_dirs(logdir)
    check_dirs(redir)
    print('peko')
    logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(logdir, args.log),
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)
    cvd = Cutvideos(metafile, args.workdir, args.resultfile, args.rm_tmp_file)
    cvd.extract_all_clip()