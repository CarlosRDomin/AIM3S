import cv2
import numpy as np
from openpose import pyopenpose as op
from datetime import datetime
from multiprocessing import Pool, cpu_count
from enum import Enum
import glob
import argparse
import h5py
import json
import os
import time

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
HDF5_FRAME_NAME_FORMAT = "frame{:05d}"
HDF5_POSE_GROUP_NAME = "pose"
CROPPED_IMGS_FOLDER_NAME = "Cropped hands"


class JointEnum(Enum):
    NOSE = 0
    NECK = 1
    RSHOULDER = 2
    RELBOW = 3
    RWRIST = 4
    LSHOULDER = 5
    LELBOW = 6
    LWRIST = 7
    MIDHIP = 8
    RHIP = 9
    RKNEE = 10
    RANKLE = 11
    LHIP = 12
    LKNEE = 13
    LANKLE = 14
    REYE = 15
    LEYE = 16
    REAR = 17
    LEAR = 18
    LBIGTOE = 19
    LSMALLTOE = 20
    LHEEL = 21
    RBIGTOE = 22
    RSMALLTOE = 23
    RHEEL = 24
    BACKGND = 25


def _ensure_folder_exists(folder):
    try:
        os.makedirs(folder)
    except OSError:  # Already exists -> Ignore
        pass


def preprocess_weight(parent_folder):
    print("Processing weights at {}".format(parent_folder))
    time.sleep(3)


def preprocess_vision(video_filename, wrist_thresh=0.2, crop_half_w=100, crop_half_h=100):
    print("Processing video '{}'...".format(video_filename))
    video_prefix = os.path.splitext(video_filename)[0]  # Remove extension
    pose_prefix = video_prefix + "_pose"
    cropped_img_prefix = os.path.join(os.path.dirname(video_prefix), CROPPED_IMGS_FOLDER_NAME, os.path.basename(video_prefix))
    _ensure_folder_exists(os.path.dirname(cropped_img_prefix))  # Create folder if it didn't exist

    # Run Openpose to find people and their poses
    if os.path.exists(pose_prefix):
        print("Folder '{}' exists, not running Openpose!".format(pose_prefix))
    else:
        openpose_params = {
            "model_folder": "openpose-models/",
            "video": video_filename,
            "write_video": pose_prefix + ".mp4",
            "write_json": pose_prefix,  # Will create the folder and save a json per frame in the video
            "display": 0,
            "logging_level": 3,
            "render_pose": 1,  # 1 for CPU (slightly faster), 2 for GPU
        }
        openpose_wrapper = op.WrapperPython(3)
        openpose_wrapper.configure(openpose_params)
        openpose_wrapper.execute()  # Blocking call
        print("Openpose done processing video '{}'!".format(video_filename))

    # Postprocess json files (one per frame), combine into a single hdf file and crop image around hands
    video_orig = cv2.VideoCapture(video_filename)
    with h5py.File(video_prefix + ".h5", 'a') as f_hdf5:
        if HDF5_POSE_GROUP_NAME in f_hdf5: del f_hdf5[HDF5_POSE_GROUP_NAME]  # OVERWRITE (delete if already existed)
        pose = f_hdf5.create_group("pose")

        # Parse every json
        for frame_i,json_filename in enumerate(sorted(os.listdir(pose_prefix))):
            frame_i_str = HDF5_FRAME_NAME_FORMAT.format(frame_i+1)
            _, frame_img = video_orig.read()

            # Parse frame json
            with open(os.path.join(pose_prefix, json_filename)) as f_json:
                data = json.load(f_json)

            # Parse pose for each person found
            n_hands = 0
            poses = []
            for p in data["people"]:
                keypoints = np.reshape(p["pose_keypoints_2d"], (-1,3))
                poses.append(keypoints)

                # Look for hands with high enough confidence and crop an image around each one
                for i_wrist in (JointEnum.LWRIST.value, JointEnum.RWRIST.value):
                    if keypoints[i_wrist,-1] > wrist_thresh:  # Found a wrist with high enough confidence
                        n_hands += 1
                        crop_img = _crop_image(frame_img, center=keypoints[i_wrist, 1::-1], half_w=crop_half_w, half_h=crop_half_h)
                        cv2.imwrite("{}_{}_{:02d}.jpg".format(cropped_img_prefix, frame_i_str, n_hands), crop_img)
            pose.create_dataset(frame_i_str, data=poses)

    print("Done processing video '{}'!".format(video_filename))


def _crop_image(img, center, half_w, half_h):
    def _min(a,b): return a if a<b else b
    def _max(a,b): return a if a>b else b

    center_x = int(center[1])
    center_y = int(center[0])
    x_min = _max(center_x-half_w, 0)
    x_max = _min(center_x+half_w, img.shape[1]-1)
    y_min = _max(center_y-half_h, 0)
    y_max = _min(center_y+half_h, img.shape[0]-1)
    return img[y_min:y_max+1, x_min:x_max+1, :]


class ExperimentPreProcessor:
    def __init__(self, main_folder, start_datetime=datetime.min, end_datetime=datetime.max, do_pose=True, do_weight=True, num_processes_weight=cpu_count(), num_processes_vision=3):
        self.main_folder = main_folder
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.do_pose = do_pose
        self.do_weight = do_weight

        self.pool_weight = Pool(processes=num_processes_weight)
        self.pool_vision = Pool(processes=num_processes_vision)
        self.weight_tasks_state = []
        self.vision_tasks_state = []
        self.num_weight_tasks_done = 0
        self.num_vision_tasks_done = 0

    @staticmethod
    def _list_subfolders(folder):
        return next(os.walk(folder))[1]

    def _task_done_cb(self, is_weight):
        if is_weight:
            self.num_weight_tasks_done += 1
            n = self.num_weight_tasks_done
            total = len(self.weight_tasks_state)
            str_type = "Weight"
        else:
            self.num_vision_tasks_done += 1
            n = self.num_vision_tasks_done
            total = len(self.vision_tasks_state)
            str_type = "Vision"

        print("{} tasks done: {}/{} ({:5.2f}%)".format(str_type, n, total, 100*n/total))

    def run(self):
        # Traverse all subfolders inside args.folder and dispatch tasks to the pool of workers
        for f in self._list_subfolders(self.main_folder):
            t = datetime.strptime(f, DATETIME_FORMAT)  # Folder name specifies the date -> Convert to datetime

            # Filter by experiment date (only consider experiments within t_start and t_end)
            if t_start <= t <= t_end:
                parent_folder = os.path.join(args.folder, f)

                # Tell the weight preprocessor to merge all weight sensors into a single h5 file
                if args.do_weight:
                    task_state = self.pool_weight.apply_async(preprocess_weight, (parent_folder,), callback=lambda _: self._task_done_cb(is_weight=True))
                    self.weight_tasks_state.append(task_state)

                # Tell the pose preprocessor to run pose estimation on every camera video
                if args.do_pose:
                    for video in glob.glob(os.path.join(parent_folder, "cam*_{}.mp4".format(f))):
                        kwds = {"crop_half_w": 200, "crop_half_h": 200} if os.path.basename(video).startswith("cam4") else {}  # Top-down camera is closer -> Crop bigger window
                        task_state = self.pool_vision.apply_async(preprocess_vision, (video,), kwds, callback=lambda _: self._task_done_cb(is_weight=False))
                        self.vision_tasks_state.append(task_state)

        print("Preprocessing weight tasks enqueued, waiting for them to complete!")
        for tasks_state in (self.weight_tasks_state, self.vision_tasks_state):
            for i,task_state in enumerate(tasks_state):
                task_state.wait()
                if not task_state.successful():
                    print("Uh oh... {} task {}: {}".format("Weight" if tasks_state==self.weight_tasks_state else "Vision", i+1, task_state._value))

        print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", default="Dataset/Characterization", help="Folder containing the experiment(s) to preprocess")
    parser.add_argument("-s", "--start-datetime", default="", help="Only preprocess experiments collected later than this datetime (format: {}; empty for no limit)".format(DATETIME_FORMAT))
    parser.add_argument("-e", "--end-datetime", default="", help="Only preprocess experiments collected before this datetime (format: {}; empty for no limit)".format(DATETIME_FORMAT))
    parser.add_argument('-p', "--do-pose", default=True, type=bool, help="Whether or not to pre-process human pose")
    parser.add_argument('-w', "--do-weight", default=True, type=bool, help="Whether or not to pre-process weight")
    parser.add_argument('-nw', "--num-processes-weight", default=cpu_count(), type=int, help="Number of processes to spawn for weight preprocessing")
    parser.add_argument('-nv', "--num-processes-vision", default=3, type=int, help="Number of processes to spawn for vision preprocessing")
    args = parser.parse_args()

    t_start = datetime.strptime(args.start_datetime, DATETIME_FORMAT) if len(args.start_datetime) > 0 else datetime.min
    t_end = datetime.strptime(args.end_datetime, DATETIME_FORMAT) if len(args.end_datetime) > 0 else datetime.max

    ExperimentPreProcessor(args.folder, t_start, t_end, args.do_pose, args.do_weight, args.num_processes_weight, args.num_processes_vision).run()
