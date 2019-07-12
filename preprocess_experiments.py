import cv2
import numpy as np
from aux_tools import str2bool, _min, _max, ensure_folder_exists, format_axis_as_timedelta, JointEnum, save_datetime_to_h5, ExperimentTraverser, EXPERIMENT_DATETIME_STR_FORMAT
from datetime import datetime
from multiprocessing import Pool, cpu_count
import traceback
import glob
import argparse
import h5py
import json
import os


class BackgroundSubtractor:
    def __init__(self):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
        self.fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def runGMG(self, frame):
        mask = self.fgbg3.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return mask

    def runMOG(self, frame):
        return self.fgbg2.apply(frame)

    def runGSOC(self, frame):
        return self.fgbg.apply(frame)

    def run(self, frame):
        return self.runGSOC(frame)


HDF5_FRAME_NAME_FORMAT = "frame{:05d}"
HDF5_POSE_GROUP_NAME = "pose"
HDF5_HANDS_GROUP_NAME = "hands"
HDF5_ORIG_WEIGHT_GROUP_NAME = "orig_weights"
HDF5_WEIGHT_GROUP_NAME = "weight_{}"
HDF5_WEIGHT_T_NAME = "t"
HDF5_WEIGHT_DATA_NAME = "w"
BACKGROUND_MASKS_FOLDER_NAME = "background_masks"


def preprocess_weight(parent_folder, do_tare=False, visualize=False):
    from read_dataset import read_weights_data
    from matplotlib import pyplot as plt

    print("Processing weights at {}".format(parent_folder))
    t_start = os.path.basename(parent_folder)

    h5_filename = os.path.join(parent_folder, "weights_{}.h5".format(t_start))
    if os.path.exists(h5_filename):
        print("File {} exists, not preprocessing!".format(h5_filename))
        return

    weight_t, weight_data, weights_orig = read_weights_data(parent_folder, do_tare=do_tare)
    with h5py.File(h5_filename, 'w') as f_hdf5:
        save_datetime_to_h5(weight_t, f_hdf5, HDF5_WEIGHT_T_NAME)
        f_hdf5.create_dataset("w", data=weight_data)

        # Save original weight info as well, just in case
        orig_weights_group = f_hdf5.create_group(HDF5_ORIG_WEIGHT_GROUP_NAME)
        for weight_id, weight_info in weights_orig.items():
            orig_weight = orig_weights_group.create_group(HDF5_WEIGHT_GROUP_NAME.format(weight_id))
            save_datetime_to_h5(weight_info['t'], orig_weight, HDF5_WEIGHT_T_NAME)
            orig_weight.create_dataset(HDF5_WEIGHT_DATA_NAME, data=weight_info['w'])

            if visualize:
                fig = plt.figure(figsize=(4, 2))
                ax = fig.subplots()
                ax.plot([(t - weight_t[0]).total_seconds() for t in weight_t], weight_data)
                ax.set_title('Load cell #{}'.format(weight_id))
                ax.set_ylabel('Weight (g)')
                format_axis_as_timedelta(ax.xaxis)
                fig.show()

    print("Done processing weights as '{}'! t_min={}; t_max={}; N={}".format(h5_filename, weight_t[0], weight_t[-1], weight_data.shape))


def preprocess_vision_object_detection(video_filename, gpu_id, config_file="configs/aim3s.yaml", confidence_thresh=0.1, categories_file="../aim3s_dataset/aim3s.names", generate_video=True):
    from maskrcnn_benchmark.config import cfg
    from predictor_skus import SKUsDemo

    video_prefix = os.path.splitext(video_filename)[0]  # Remove extension
    file_prefix = video_prefix + "_objdet"

    # Load MaskRCNN config
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = gpu_id  # Run the model on the specified gpu
    #cfg.freeze()

    # Load category names
    with open(categories_file) as f:
        categories = f.readlines()

    # Prepare object that handles inference plus adds predictions on top of image
    model = SKUsDemo(
        cfg,
        categories=categories,
        confidence_threshold=confidence_thresh,
    )

    # Initialize video
    v = cv2.VideoCapture(video_filename)
    N = v.get(cv2.CAP_PROP_FRAME_COUNT)
    n = 0
    if generate_video:
        v_out = cv2.VideoWriter("{}.mp4".format(file_prefix), cv2.VideoWriter_fourcc(*'mp4v'), 25.0,
            (int(v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Process video
    with h5py.File("{}.h5".format(file_prefix), 'w') as f:
        while n < N:  # Read every frame
            images = []
            while n < N and len(images) < 15:  # Read a block of frames
                ok, img = v.read()
                assert ok, "Couldn't read frame from video {}".format(video_filename)
                images.append(img)
                n += 1

            # Process frame block
            preds = model.compute_prediction_list(images)

            for i, predictions in enumerate(preds):
                predictions = model.select_top_predictions(predictions)
                f.create_dataset(HDF5_FRAME_NAME_FORMAT.format(n-len(preds)+i+1), data=predictions.get_field("scores_all").numpy())

                if generate_video:
                    img = model.overlay_boxes(images[i], predictions)
                    img = model.overlay_class_names(img, predictions)
                    v_out.write(img)

                # Display progress
                #if n % 25 == 0:
            print("Processed frame {}/{} for video {} ({:.2f}%)".format(n, N, video_filename, 100.*n/N))

    if generate_video:
        v_out.release()


def preprocess_vision(video_filename, pose_model_folder, wrist_thresh=0.2, crop_half_w=100, crop_half_h=100):
    print("Processing video '{}'...".format(video_filename))
    video_prefix = os.path.splitext(video_filename)[0]  # Remove extension
    pose_prefix = video_prefix + "_pose"
    mask_prefix = os.path.join(os.path.dirname(video_prefix), BACKGROUND_MASKS_FOLDER_NAME, os.path.basename(video_prefix) + "_mask")
    ensure_folder_exists(os.path.dirname(mask_prefix))  # Create folder if it didn't exist

    # Run Openpose to find people and their poses
    if os.path.exists(pose_prefix) and len(os.listdir(pose_prefix)) > 0:
        print("Folder '{}' exists, not running Openpose!".format(pose_prefix))
    else:
        from openpose import pyopenpose as op
        openpose_params = {
            "model_folder": pose_model_folder,
            "video": video_filename,
            "write_video": pose_prefix + ".mp4",
            "write_json": pose_prefix,  # Will create the folder and save a json per frame in the video
            "display": 0,
            "render_pose": 1,  # 1 for CPU (slightly faster), 2 for GPU
        }
        openpose_wrapper = op.WrapperPython(3)
        openpose_wrapper.configure(openpose_params)
        openpose_wrapper.execute()  # Blocking call
        print("Openpose done processing video '{}'!".format(video_filename))

    # Initialize background subtractor
    video_orig = cv2.VideoCapture(video_filename)
    video_mask = cv2.VideoWriter("{}_mask.mp4".format(video_prefix), cv2.VideoWriter_fourcc(*'avc1'), 25.0,
            (int(video_orig.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    bgnd_subtractor = BackgroundSubtractor()

    # Postprocess json files (one per frame) + combine into a single hdf file, as well as compute bgnd subtraction mask
    with h5py.File(video_prefix + ".h5", 'a') as f_hdf5:
        if HDF5_POSE_GROUP_NAME in f_hdf5: del f_hdf5[HDF5_POSE_GROUP_NAME]  # OVERWRITE (delete if already existed)
        if HDF5_HANDS_GROUP_NAME in f_hdf5: del f_hdf5[HDF5_HANDS_GROUP_NAME]
        pose = f_hdf5.create_group(HDF5_POSE_GROUP_NAME)
        hands = f_hdf5.create_group(HDF5_HANDS_GROUP_NAME)

        # Parse every json (every frame of video_orig)
        for frame_i,json_filename in enumerate(sorted(os.listdir(pose_prefix))):
            frame_i_str = HDF5_FRAME_NAME_FORMAT.format(frame_i+1)
            _, frame_img = video_orig.read()

            # Run background subtractor
            background_mask = bgnd_subtractor.run(frame_img)
            background_removed_img = cv2.bitwise_and(frame_img, frame_img, mask=background_mask)
            video_mask.write(background_removed_img)

            # Parse frame json
            with open(os.path.join(pose_prefix, json_filename)) as f_json:
                data = json.load(f_json)

            # Parse pose for each person found
            hands_info = []
            poses = []
            for i_person,p in enumerate(data["people"]):
                keypoints = np.reshape(p["pose_keypoints_2d"], (-1,3))
                poses.append(keypoints)

                # Look for hands with high enough confidence and crop an image around each one
                for i_wrist in (JointEnum.LWRIST.value, JointEnum.RWRIST.value):
                    if keypoints[i_wrist,-1] > wrist_thresh:  # Found a wrist with high enough confidence
                        center = keypoints[i_wrist, 0:2]
                        hands_info.append(np.hstack((center, i_person, i_wrist)))  # [x, y, person_id, wrist_id] (wrist_id see JointEnum, 4=Right;7=Left)
            pose.create_dataset(frame_i_str, data=poses)
            hands.create_dataset(frame_i_str, data=hands_info)
            cv2.imwrite("{}_{}.png".format(mask_prefix, frame_i_str), background_mask)

    video_mask.release()
    print("Done processing video '{}'!".format(video_filename))


def _crop_image(img, center, half_w, half_h):
    center_x = int(center[0])
    center_y = int(center[1])
    x_min = _max(center_x-half_w, 0)
    x_max = _min(center_x+half_w, img.shape[1]-1)
    y_min = _max(center_y-half_h, 0)
    y_max = _min(center_y+half_h, img.shape[0]-1)
    return img[y_min:y_max+1, x_min:x_max+1, :]


class ExperimentPreProcessor(ExperimentTraverser):
    def __init__(self, main_folder, start_datetime=datetime.min, end_datetime=datetime.max, do_weight=True, do_pose=True, do_objdet=True, pose_model_folder="openpose-models/", num_processes_weight=cpu_count(), num_processes_vision=3, num_processes_objdet=4, num_gpus=3):
        super(ExperimentPreProcessor, self).__init__(main_folder, start_datetime, end_datetime)
        self.do_weight = do_weight
        self.do_objdet = do_objdet
        self.do_pose = do_pose
        self.pose_model_folder = pose_model_folder

        self.pool_weight = Pool(processes=num_processes_weight) if do_weight else None
        self.pool_vision = Pool(processes=num_processes_vision) if do_pose else None
        self.pool_objdet = [Pool(processes=num_processes_objdet) if do_objdet else None for i in range(num_gpus)]
        self.weight_tasks_state = []
        self.vision_tasks_state = []
        self.num_weight_tasks_done = 0
        self.num_vision_tasks_done = 0
        self.next_gpu = 0
        self.num_gpus = num_gpus

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

    def process_subfolder(self, f):
        parent_folder = os.path.join(self.main_folder, f)

        # Tell the weight preprocessor to merge all weight sensors into a single h5 file
        if self.do_weight:
            task_state = self.pool_weight.apply_async(preprocess_weight, (parent_folder,), callback=lambda _: self._task_done_cb(is_weight=True))
            self.weight_tasks_state.append(task_state)

        # Tell the pose preprocessor to run pose estimation on every camera video
        for video in glob.glob(os.path.join(parent_folder, "cam*_{}.mp4".format(f))):
            if self.do_pose:
                kwds = {"crop_half_w": 200, "crop_half_h": 200} if os.path.basename(video).startswith("cam4") else {}  # Top-down camera is closer -> Crop bigger window
                task_state = self.pool_vision.apply_async(preprocess_vision, (video, self.pose_model_folder), kwds, callback=lambda _: self._task_done_cb(is_weight=False))
                self.vision_tasks_state.append(task_state)

            if self.do_objdet:
                task_state = self.pool_objdet[self.next_gpu].apply_async(preprocess_vision_object_detection, (video, self.next_gpu), callback=lambda _: self._task_done_cb(is_weight=False))
                self.next_gpu = (self.next_gpu+1) % self.num_gpus
                self.vision_tasks_state.append(task_state)


    def on_done(self):
        print("Preprocessing tasks enqueued, waiting for them to complete!")
        for tasks_state in (self.weight_tasks_state, self.vision_tasks_state):
            for i,task_state in enumerate(tasks_state):
                try:
                    task_state.get()
                except Exception as e:
                    traceback.print_exc()
                # task_state.wait()
                # if not task_state.successful():
                #     print("Uh oh... {} task {}: {}".format("Weight" if tasks_state==self.weight_tasks_state else "Vision", i+1, task_state._value))

        print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", default="Dataset/Evaluation", help="Folder containing the experiment(s) to preprocess")
    parser.add_argument("-s", "--start-datetime", default="", help="Only preprocess experiments collected later than this datetime (format: {}; empty for no limit)".format(EXPERIMENT_DATETIME_STR_FORMAT))
    parser.add_argument("-e", "--end-datetime", default="", help="Only preprocess experiments collected before this datetime (format: {}; empty for no limit)".format(EXPERIMENT_DATETIME_STR_FORMAT))
    parser.add_argument('-w', "--do-weight", default=True, type=str2bool, help="Whether or not to pre-process weight")
    parser.add_argument('-p', "--do-pose", default=True, type=str2bool, help="Whether or not to pre-process human pose")
    parser.add_argument('-o', "--do-objdet", default=True, type=str2bool, help="Whether or not to pre-process videos with object detection")
    parser.add_argument('-pm', "--pose-model-folder", default="openpose-models/", help="Human pose model folder location (can be a symlink)")
    parser.add_argument('-nw', "--num-processes-weight", default=cpu_count(), type=int, help="Number of processes to spawn for weight preprocessing")
    parser.add_argument('-nv', "--num-processes-vision", default=3, type=int, help="Number of processes to spawn for vision preprocessing")
    parser.add_argument('-no', "--num-processes-objdet", default=4, type=int, help="Number of processes to spawn for object detection preprocessing (will be multiplied by the number of GPUs)")
    parser.add_argument('-ng', "--num-gpus", default=1, type=int, help="Number of GPUs available")
    args = parser.parse_args()

    t_start = datetime.strptime(args.start_datetime, EXPERIMENT_DATETIME_STR_FORMAT) if len(args.start_datetime) > 0 else datetime.min
    t_end = datetime.strptime(args.end_datetime, EXPERIMENT_DATETIME_STR_FORMAT) if len(args.end_datetime) > 0 else datetime.max

    ExperimentPreProcessor(args.folder, t_start, t_end, args.do_weight, args.do_pose, args.do_objdet, args.pose_model_folder, args.num_processes_weight, args.num_processes_vision, args.num_processes_objdet, args.num_gpus).run()

