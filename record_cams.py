import cv2
from datetime import datetime,timedelta
from multiprocessing import Process, Pipe
import argparse
import os
import h5py
from random import random
from aux_tools import get_nonempty_input


class ProcessRecordCam:
    """
     Helper class that allows recording a camera stream *in a separate process*
    """
    ALPHA_FPS = 0.95
    ALIVE_PRINT_T = timedelta(seconds=2)  # Print a message every 2s

    class FakeVideoWriter(object):
        def write(self, img):
            pass

        def release(self):
            pass

    def __init__(self, rtsp_ip, rtsp_user="admin", rtsp_pass="", rtsp_ch="1", rtsp_stream="0", out_filename=None, out_codec="H264", out_fps=30, cam_id=1, recording_info=None, visualize=False):
        self.rtsp_ip = rtsp_ip
        self.rtsp_user = rtsp_user
        self.rtsp_pass = rtsp_pass
        self.rtsp_ch = rtsp_ch
        self.rtsp_stream = rtsp_stream
        self.out_filename = out_filename
        self.out_codec = out_codec
        self.out_fps = out_fps
        self.cam_id = "{} ({})".format(cam_id, self.rtsp_ip)
        self.recording_info = recording_info if recording_info is not None else {}
        self.visualize = visualize

        self.video_src = "rtsp://{}/user={}&password={}&channel={}&stream={}.sdp".format(rtsp_ip, rtsp_user, rtsp_pass, rtsp_ch, rtsp_stream)
        self.fps = 0
        self.t_frames = []
        self.t_alive_next_print = datetime.now() + timedelta(seconds=random()) + self.ALIVE_PRINT_T

    def __call__(self, pipe):
        try:
            self.pipe = pipe

            # Initialize video input
            self.video_in = cv2.VideoCapture(self.video_src)

            # Initialize visualization
            if self.visualize:
                self.win_name = "Live feed @{}".format(self.rtsp_ip)
                cv2.namedWindow(self.win_name)
                cv2.waitKeyEx(1)

            # Initialize output video
            ok, img = self.video_in.read()
            assert ok, "Couldn't read from camera {} (url: {})".format(self.cam_id, self.video_src)
            print("Camera {} initialized!".format(self.cam_id))
            self.video_out = cv2.VideoWriter(self.out_filename, cv2.VideoWriter_fourcc(*self.out_codec), self.out_fps, img.shape[1::-1]) if self.out_filename is not None else ProcessRecordCam.FakeVideoWriter()

            while True:
                # Check for a request to stop recording
                if self.pipe.poll():
                    command = self.pipe.recv()
                    break

                # Fetch image
                ok, img = self.video_in.read()
                self.t_frames.append(datetime.now())
                self.video_out.write(img)

                # Render
                if self.visualize:
                    if len(self.t_frames) > 1:
                        self.fps = self.ALPHA_FPS*self.fps + (1-self.ALPHA_FPS)/(self.t_frames[-1]-self.t_frames[-2]).total_seconds()
                        cv2.putText(img, "FPS: {:5.2f}".format(self.fps), (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))
                    cv2.imshow(self.win_name, img)
                    key = cv2.waitKeyEx(1)
                    if key == ord(' '):
                        cv2.imwrite(os.path.join(os.path.dirname(self.out_filename), "screenshot.jpg"), img)
                        break

                # Print alive message if necessary
                if self.t_frames[-1] > self.t_alive_next_print:
                    self.t_alive_next_print += self.ALIVE_PRINT_T
                    print("@{} - Camera {} is alive".format(str(self.t_frames[-1])[:-3], self.cam_id))
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:  # Print a message on KeyboardInterrupt as well as exit request through self.pipe
            print("Exit requested for camera {}".format(self.cam_id))

        # Clean up
        print("Closing camera {}...".format(self.cam_id))
        self.video_out.release()
        self._log_config()
        if self.visualize:
            cv2.destroyAllWindows()
        print("Closed camera {}, video saved as '{}'!".format(self.cam_id, self.out_filename))

    def _log_config(self):
        with h5py.File(os.path.splitext(self.out_filename)[0] + ".h5", 'w') as f:
            f.create_dataset("t", data=[(t-self.t_frames[0]).total_seconds() for t in self.t_frames])
            f.create_dataset("t_str", data=[str(t).encode('utf8') for t in self.t_frames])
            config = f.create_group("config")
            config.attrs["ip"] = self.rtsp_ip
            config.attrs["channel"] = self.rtsp_ch
            config.attrs["stream"] = self.rtsp_stream
            config.attrs["out_filename"] = self.out_filename
            config.attrs["cam_id"] = self.cam_id
            for info_key, info_value in self.recording_info.items():
                config.attrs[info_key] = info_value


class ProcessRecordCamHelper:
    """
     Helper class responsible for creating and communicating with a ProcessRecordCam instance.
     That is: it spawns a new process that runs ProcessRecordCam, starts it, and and allows stopping the cam recording
    """

    def __init__(self, *args, **kwargs):
        self.pipe, record_cam_pipe = Pipe()  # Create a Pipe to communicate both processes
        self.cam_recorder = ProcessRecordCam(*args, **kwargs)  # Forward args (and kwargs) to ProcessRecordCam
        self.cam_recorder_process = Process(target=self.cam_recorder, args=(record_cam_pipe,))  # Spawn a new process and run the ProcessRecordCam in it
        self.cam_recorder_process.daemon = True  # Kill the process when the main process is killed
        self.cam_recorder_process.start()

    def stop_recording(self):
        self.pipe.send("EXIT")

    def wait_for_finish(self):
        return self.cam_recorder_process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ip", nargs='+', help="IP of the camera")
    parser.add_argument('-u', "--user", default="admin", help="Camera's username")
    parser.add_argument('-p', "--pass", dest="passwd", default="", help="Camera's password")
    parser.add_argument('-c', "--channel", dest="ch", default="1", help="Camera's channel")
    parser.add_argument('-s', "--stream", default="0", help="Camera's stream")
    parser.add_argument('-f', "--folder", default="Dataset/Evaluation", help="Recording output folder")
    parser.add_argument('-k', "--codec", default="avc1", help="Output video codec")
    parser.add_argument('-r', "--fps", default=30, type=int, help="Output video frame rate")
    parser.add_argument('-m', "--multiple-recordings", default=False, action="store_true", help="Append this flag to allow for multiple recordings without needing to re-run the script")
    parser.add_argument('-v', "--visualize", default=False, action="store_true", help="Append this flag to visualize the camera(s) in an OpenCV window")
    args = parser.parse_args()

    while True:
        recording_info = {}
        if args.multiple_recordings:
            item_name = get_nonempty_input("Please enter the name of this item once it's placed in the center of the turntable (or enter 'QUIT' to terminate): ")
            if item_name.lower() == "quit":
                break
            item_barcode = get_nonempty_input("Input the item's barcode: ")
            item_frontback = get_nonempty_input("Is this the front (f) or the back (b)? ")
            recording_info = {"item_name": item_name, "item_barcode": item_barcode, "item_frontback": item_frontback}
            raw_input("Ready? Press ENTER")

        t_start = str(datetime.now())[:-7].replace(':', '-').replace(' ', '_')
        print("Starting cam recording @ {}".format(t_start))
        video_suffix = t_start if not args.multiple_recordings else '_'.join((item_name, item_barcode, item_frontback))
        output_folder = os.path.join(args.folder, video_suffix)
        os.makedirs(output_folder)  # Create parent directory

        p_recordings = []
        for i,ip in enumerate(args.ip):
            p_recordings.append(ProcessRecordCamHelper(ip, args.user, args.passwd, args.ch, args.stream, os.path.join(output_folder, "cam{}_{}.mp4".format(i+1, video_suffix)), args.codec, args.fps, i+1, recording_info, args.visualize))

        try:
            print("\nPress Ctrl+C {}\n".format("after this item has spinned back and fourth at least twice" if args.multiple_recordings else "to stop recording"))
            for p in p_recordings: p.wait_for_finish()
        except (KeyboardInterrupt, SystemExit):  # Wait for user to request to stop recording
            for p in p_recordings: p.stop_recording()
            for p in p_recordings: p.wait_for_finish()

        if not args.multiple_recordings:  # Only want to record once, exit
            break

    print("Goodbye! :)")
