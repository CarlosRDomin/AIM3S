from read_dataset import read_weight_data
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import h5py
from datetime import datetime, timedelta


def plt_fig_to_cv2_img(fig):
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr

    return img

def generate_video(experiment_base_folder='Dataset/Characterization/2019-03-31_00-00-02', camera_id=3, weight_id=5309446, t_lims=5, weight_plot_scale=0.3):
    # Read all weight sensors for the full experiment duration at once
    weight_t, weight_data = read_weight_data(os.path.join(experiment_base_folder, 'sensors_{}'.format(weight_id)))

    # Set up camera files to read
    t_experiment_start = experiment_base_folder.rsplit('/', 1)[-1]  # Last folder in the path should indicate time at which experiment started
    camera_filename = os.path.join(experiment_base_folder, "cam{}_{}".format(camera_id, t_experiment_start))
    video_in = cv2.VideoCapture(camera_filename + ".mp4")
    camera_info = h5py.File(camera_filename + ".h5", 'r')
    camera_timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in camera_info.get("t_str")]

    # Manually align weight and cam timestamps (not synced for some reason)
    weight_to_cam_t_offset = camera_timestamps[0] - timedelta(seconds=2)  # - weight_t[0]

    # Set up matplotlib figure
    fig = plt.figure(figsize=(4,2))
    ax = fig.subplots()
    ax.plot(weight_t - weight_t[0], weight_data)
    ax.set_title('Load cell #{}'.format(weight_id))
    ax.set_ylabel('Weight (g)')

    # Set up video file
    video_filename = 'AIM3S_experiment.mp4'
    video_fps = 30
    video_out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'avc1'), video_fps, (int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    for n,t in enumerate(camera_timestamps):
        ok, rgb_data = video_in.read()
        assert ok, "Couldn't read frame {}!".format(n)

        # Update weight plot and convert to image
        curr_t = (t-weight_to_cam_t_offset).total_seconds()
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
    generate_video()
