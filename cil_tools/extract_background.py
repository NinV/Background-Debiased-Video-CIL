import argparse
import pathlib
from multiprocessing import Process
from tqdm import tqdm
import cv2
import imutils
import numpy as np
from typing import List
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--from_video', action='store_true', required=True)
    parser.add_argument('--image_suffix', default='.jpg')
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=500)
    parser.add_argument('--size', type=int, default=256)
    return parser.parse_args()


def short_side_resize(img):
    h, w = img.shape[:2]
    if h > w:
        img = imutils.resize(img, width=w)
    else:
        img = imutils.resize(img, height=h)
    return img


def bg_extraction_tmf(data_path: pathlib.Path, dest: pathlib.Path,
                      from_video: bool, interval: int, max_frames: int):
    """
    extract background using median temporal filtering
    https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/
    """
    frames = []
    count = 0
    if from_video:
        cap = cv2.VideoCapture(str(data_path))
        while cap.isOpened() and len(frames) <= max_frames:
            ret, frame_ = cap.read()
            if count % interval == 0:
                if ret:
                    frames.append(frame_)
                    short_side_resize(frame_)
                else:
                    break
            count += 1
    else:
        image_files = data_path.glob('*')
        for img_f in image_files:
            if len(frames) > max_frames:
                break
            if count % interval == 0:
                img = cv2.imread(str(img_f))
                if img:
                    short_side_resize(img)
                    frames.append(img)
            count += 1

    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    cv2.imwrite(str(dest), median_frame)
    return median_frame


def bg_extract_multiple(paths: List[pathlib.Path], output_dir: pathlib.Path, from_video: bool,
                        interval: int, max_frames: int, process_id: int):
    for data_path in tqdm(paths, total=len(paths),
                          desc='Extracting background #{}'.format(process_id),
                          unit='video', position=process_id, leave=False):
        bg_extraction_tmf(data_path, (output_dir / data_path.name).with_suffix('.jpg'), from_video, interval, max_frames)
        # pbar.update(1)


if __name__ == '__main__':
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    video_dir = pathlib.Path(args.video_dir)

    # check duplicated background
    video_paths = set(video_dir.glob("*"))
    extracted = []
    for p_ in video_paths:
        if (output_dir / p_.name).with_suffix(args.image_suffix).exists():
            extracted.append(p_)
    video_paths = list(video_paths.difference(extracted))
    print('Found {} backgrounds'.format(len(extracted)))
    print('Extracting background from {} videos'.format(len(video_paths)))

    splits = []
    start = 0
    num_videos_per_process = math.ceil(len(video_paths) / args.num_workers)
    for i in range(args.num_workers):
        splits.append(video_paths[start: start + num_videos_per_process])
        start += num_videos_per_process

    # pbar = tqdm(desc='Extracting background',
    #             total=len(video_paths),
    #             unit='video'
    #             )

    processes = []
    for i in range(len(splits)):
        p = Process(target=bg_extract_multiple, args=(splits[i], output_dir, args.from_video,
                                                      args.interval, args.max_frames, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()



