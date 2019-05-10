from read_dataset import read_weight_data, read_weights_data
from aux_tools import format_axis_as_timedelta, _min, _max, str2bool, DEFAULT_TIMEZONE, date_range, time_to_float
import cv2
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import os
import h5py
import argparse


def plt_fig_to_cv2_img(fig):
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr

    return img


def generate_video(experiment_base_folder='Dataset/Characterization/2019-03-31_00-00-02', camera_id=3, weight_id=5309446, do_tare=False, t_lims=5, t_start=0, t_end=-1, weight_plot_scale=0.3, video_fps=25):
    multiple_cams = (camera_id < 0)
    if multiple_cams:
        weight_plot_scale = 1  # Overwrite setting, weight plots will be hstacked (same height)

    # Read all weight sensors for the full experiment duration at once
    multiple_weights = (weight_id < 0)
    if multiple_weights:
        weight_t, weight_data, _ = read_weights_data(experiment_base_folder)
        w = np.sum(weight_data, axis=1)
    else:
        weight_t, weight_data, _ = read_weight_data(os.path.join(experiment_base_folder, 'sensors_{}'.format(weight_id)), do_tare=do_tare)
        w = [weight_data]
    t_w = time_to_float(weight_t, weight_t[0])

    # Set up camera files to read
    t_experiment_start = experiment_base_folder.rsplit('/', 1)[-1]  # Last folder in the path should indicate time at which experiment started
    video_in = []
    camera_timestamps = []
    t_latest_start = datetime.min.replace(tzinfo=DEFAULT_TIMEZONE)
    t_earliest_end = datetime.max.replace(tzinfo=DEFAULT_TIMEZONE)
    for cam in (range(4) if multiple_cams else [camera_id]):
        camera_filename = os.path.join(experiment_base_folder, "cam{}_{}".format(cam+1, t_experiment_start))
        video_in.append(cv2.VideoCapture(camera_filename + ".mp4"))
        camera_info = h5py.File(camera_filename + ".h5", 'r')
        camera_timestamps.append(np.array([DEFAULT_TIMEZONE.localize(datetime.strptime(t.decode('utf8'), "%Y-%m-%d %H:%M:%S.%f")) for t in camera_info.get("t_str")]))
        t_latest_start = _max(camera_timestamps[-1][0], t_latest_start)
        t_earliest_end = _min(camera_timestamps[-1][-1], t_earliest_end)

    # Interpolate time (nearest frame) as if cameras had been sampled at constant fps
    t_cam = np.array(list(date_range(t_latest_start, t_earliest_end, timedelta(seconds=1/video_fps))))
    to_float = lambda t_arr: np.array(time_to_float(t_arr, t_latest_start))
    frame_nums = []
    for t in camera_timestamps:
        frame_nums.append(interp1d(to_float(t), range(len(t)), kind='nearest', copy=False, assume_sorted=True)(to_float(t_cam)).astype(np.uint16))
    frame_nums = np.array(frame_nums)

    # Manually align weight and cam timestamps (not synced for some reason)
    weight_to_cam_t_offset = weight_t[0] + timedelta(seconds=13)  # camera_timestamps[0]

    # Set up matplotlib figure
    fig = plt.figure(figsize=(3.5,5) if multiple_cams else (4,2))
    num_axes = len(weight_data) if multiple_weights else 1
    ax = fig.subplots(num_axes, 1, sharex=True, squeeze=False)
    for i in range(num_axes):
        shelf_i = num_axes - (i+1)  # Shelf 1 is at the bottom
        ax[i,0].plot(t_w, w[shelf_i])
        ax[i,0].set_title('Shelf {}'.format(shelf_i+1) if multiple_weights else 'Load cell #{}'.format(weight_id))
        ax[i,0].set_ylabel('Weight (g)')
        format_axis_as_timedelta(ax[i,0].xaxis)

    # Set up video file
    video_filename = 'AIM3S_experiment.mp4'
    video_out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'avc1'), video_fps, (int(video_in[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_in[0].get(cv2.CAP_PROP_FRAME_HEIGHT))))

    for n,t in enumerate(t_cam):
        imgs = []
        for i in range(len(frame_nums)):
            video_in[i].set(cv2.CAP_PROP_POS_FRAMES, frame_nums[i,n])
            ok, img = video_in[i].read()
            assert ok, "Couldn't read frame {} from camera {}!".format(n, i+1 if multiple_cams else camera_id)
            imgs.append(img if not multiple_cams else cv2.resize(img, None, fx=0.5, fy=0.5))
        rgb_data = imgs[0] if not multiple_cams else np.hstack((np.vstack(imgs[0:2]), np.vstack(imgs[2:4])))

        # Update weight plot and convert to image
        curr_t = (t-weight_to_cam_t_offset).total_seconds()
        if curr_t < t_start or (t_end > 0 and curr_t > t_end): continue
        ax[-1,0].set_xlim(curr_t-t_lims, curr_t+t_lims)
        fig.canvas.draw()
        weight_img = plt_fig_to_cv2_img(fig)

        # Rescale weight (make the plot occupy weight_plot_scale of the whole camera frame)
        scale = weight_plot_scale*rgb_data.shape[0]/weight_img.shape[0]
        weight_img = cv2.resize(weight_img, None, fx=scale, fy=scale)

        if multiple_cams:  # Place the weight plot on the right of the camera frames
            rgb_data = np.hstack((rgb_data, weight_img))
        else:  # Overwrite bottom-right corner of RGB frame with weight plot
            rgb_data[-weight_img.shape[0]:, -weight_img.shape[1]:, :] = weight_img

        # Output the image (show it and write to file)
        cv2.imshow("Frame", rgb_data)
        video_out.write(rgb_data)
        print("{} out of {} frames ({:6.2f}%) written!".format(n+1, len(t_cam), 100.0*(n+1)/len(t_cam)))

        # Let the visualization be stopped by pressing a key
        k = cv2.waitKey(1)
        if k > 0:
            print('Key pressed, exiting!')
            break

    # Close video file
    video_out.release()  # Make sure to release the video so it's actually written to disk
    print("Video successfully saved as '{}'! :)".format(video_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", default="Dataset/Characterization", help="Folder containing the experiment to visualize")
    parser.add_argument('-c', "--cam", default=1, type=int, help="ID of the camera to visualize")
    parser.add_argument('-w', "--weight", default=5309446, type=int, help="ID of the weight sensor to visualize")
    parser.add_argument('-t', "--do-tare", default=True, type=str2bool, help="Whether or not to tare the weight scale")
    parser.add_argument('-l', "--t-lims", default=5, type=float, help="Length (in s) of the weight plot sliding window")
    parser.add_argument('-s', "--t-start", default=0, type=float, help="Experiment time at which to start generating the video")
    parser.add_argument('-e', "--t-end", default=-1, type=float, help="Experiment time at which to stop generating the video (-1 for no limit)")
    parser.add_argument('-k', "--scale", default=0.3, type=float, help="Ratio (0-1) to scale down the weight plot wrt the video's dimensions")
    parser.add_argument('-r', "--fps", default=25, type=int, help="Output video frame rate")
    args = parser.parse_args()

    generate_video(args.folder, args.cam, args.weight, args.do_tare, args.t_lims, args.t_start, args.t_end, args.scale, args.fps)
