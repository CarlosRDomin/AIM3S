from read_dataset import read_weight_data
from aux_tools import format_axis_as_timedelta, str2bool, DEFAULT_TIMEZONE
import cv2
import numpy as np
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


def generate_video(experiment_base_folder='Dataset/Characterization/2019-03-31_00-00-02', camera_id=3, weight_id=5309446, do_tare=True, t_lims=5, t_start=0, t_end=-1, weight_plot_scale=0.3, video_fps=25):
    # Read all weight sensors for the full experiment duration at once
    weight_t, weight_data, _ = read_weight_data(os.path.join(experiment_base_folder, 'sensors_{}'.format(weight_id)), do_tare=do_tare)

    # Set up camera files to read
    t_experiment_start = experiment_base_folder.rsplit('/', 1)[-1]  # Last folder in the path should indicate time at which experiment started
    camera_filename = os.path.join(experiment_base_folder, "cam{}_{}".format(camera_id, t_experiment_start))
    video_in = cv2.VideoCapture(camera_filename + ".mp4")
    camera_info = h5py.File(camera_filename + ".h5", 'r')
    camera_timestamps = [DEFAULT_TIMEZONE.localize(datetime.strptime(t.decode('utf8'), "%Y-%m-%d %H:%M:%S.%f")) for t in camera_info.get("t_str")]

    # Manually align weight and cam timestamps (not synced for some reason)
    weight_to_cam_t_offset = weight_t[0] + timedelta(seconds=13)  # camera_timestamps[0]

    # Set up matplotlib figure
    fig = plt.figure(figsize=(4,2))
    ax = fig.subplots()
    ax.plot([(t - weight_t[0]).total_seconds() for t in weight_t], weight_data)
    ax.set_title('Load cell #{}'.format(weight_id))
    ax.set_ylabel('Weight (g)')
    format_axis_as_timedelta(ax.xaxis)

    # Set up video file
    video_filename = 'AIM3S_experiment.mp4'
    video_out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'avc1'), video_fps, (int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    for n,t in enumerate(camera_timestamps):
        ok, rgb_data = video_in.read()
        assert ok, "Couldn't read frame {}!".format(n)

        # Update weight plot and convert to image
        curr_t = (t-weight_to_cam_t_offset).total_seconds()
        if curr_t < t_start or (t_end > 0 and curr_t > t_end): continue
        ax.set_xlim(curr_t-t_lims, curr_t)
        fig.canvas.draw()
        weight_img = plt_fig_to_cv2_img(fig)

        # Rescale weight (make the plot occupy weight_plot_scale of the whole camera frame)
        scale = weight_plot_scale*rgb_data.shape[1]/weight_img.shape[1]
        weight_img = cv2.resize(weight_img, None, fx=scale, fy=scale)

        # Overwrite bottom-right corner of RGB frame with weight plot
        rgb_data[-weight_img.shape[0]:, -weight_img.shape[1]:, :] = weight_img

        # Output the image (show it and write to file)
        cv2.imshow("Frame", rgb_data)
        video_out.write(rgb_data)
        print("{} out of {} frames ({:6.2f}%) written!".format(n+1, len(camera_timestamps), 100.0*(n+1)/len(camera_timestamps)))

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
    parser.add_argument('-w', "--weight", default=5309446, help="ID of the weight sensor to visualize")
    parser.add_argument('-t', "--do-tare", default=True, type=str2bool, help="Whether or not to tare the weight scale")
    parser.add_argument('-l', "--t-lims", default=5, type=float, help="Length (in s) of the weight plot sliding window")
    parser.add_argument('-s', "--t-start", default=0, type=float, help="Experiment time at which to start generating the video")
    parser.add_argument('-e', "--t-end", default=-1, type=float, help="Experiment time at which to stop generating the video (-1 for no limit)")
    parser.add_argument('-k', "--scale", default=0.3, type=float, help="Ratio (0-1) to scale down the weight plot wrt the video's dimensions")
    parser.add_argument('-r', "--fps", default=25, type=int, help="Output video frame rate")
    args = parser.parse_args()

    generate_video(args.folder, args.cam, args.weight, args.do_tare, args.t_lims, args.t_start, args.t_end, args.scale, args.fps)
