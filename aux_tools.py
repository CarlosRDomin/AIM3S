import cv2
import numpy as np
import argparse
import pytz
import os
from enum import Enum

# Python 2-3 compatibility
import sys
if sys.version_info.major < 3:
    input = raw_input


# Aux functions
def str2bool(s):
    return s.lower() not in {"no", "false", "n", "f", "0"}

def _min(a, b):
    return a if a < b else b

def _max(a, b):
    return a if a > b else b

def ensure_folder_exists(folder):
    try:
        os.makedirs(folder)
    except OSError:  # Already exists -> Ignore
        pass

def list_subfolders(folder):
    return next(os.walk(folder))[1]

def format_axis_as_timedelta(axis):  # E.g. axis=ax.xaxis
    from matplotlib import pyplot as plt
    from datetime import timedelta

    def timedelta2str(td):
        s = str(td)
        return s if not s.startswith("0:") else s[2:]

    axis.set_major_formatter(plt.FuncFormatter(lambda x, pos: timedelta2str(timedelta(seconds=x))))

def get_nonempty_input(msg):
    out = ""
    while len(out) < 1:
        out = input(msg)
    return out


# Aux constants & enums
DEFAULT_TIMEZONE = pytz.timezone('America/Los_Angeles')

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


# Aux helper classes
class HSVthreshHelper:
    WIN_NAME = "HSV thresholding aux tool"
    PIXEL_INFO_RADIUS = 3
    PIXEL_INFO_COLOR = (255, 0, 0)
    PIXEL_INFO_THICKNESS = 2

    def __init__(self, input):
        self.input = input
        self.H_min = 0
        self.H_max = 179
        self.S_min = 0
        self.S_max = 255
        self.V_min = 0
        self.V_max = 255
        self.is_playing = True  # Play/pause for videos/streams
        self.pixel = (0, 0)
        self.show_pixel_info = True

    def get_str_lims(self):
        return "H: {}-{}, S: {}-{}, V: {}-{}".format(self.H_min, self.H_max, self.S_min, self.S_max, self.V_min, self.V_max)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags&cv2.EVENT_FLAG_LBUTTON):
            self.pixel = (x, y)
            self.show_pixel_info = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.show_pixel_info = False

    def run(self):
        # Input could be an image, a camera id, or a video/IP cam. Figure out which one.
        img = cv2.imread(self.input)  # Try opening input as an image
        is_video = (img is None)  # If it didn't work, then input is a video
        if is_video:
            try:  # Try to convert input to int (camera number)
                self.input = int(self.input)
            except ValueError:
                pass  # Input is a video or an IP cam, nothing to do
            video = cv2.VideoCapture(self.input)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create a window with 6 sliders (HSV min and max)
        cv2.namedWindow(self.WIN_NAME)
        lims = (179, 255, 255)
        for i_hsv, hsv in enumerate("HSV"):
            for minmax in ("min", "max"):
                name = hsv + '_' + minmax  # e.g. H_min
                cv2.createTrackbar(name, self.WIN_NAME, getattr(self, name), lims[i_hsv]-1 if minmax=="min" else lims[i_hsv], lambda v, what=name: setattr(self, what, v))
        cv2.setMouseCallback(self.WIN_NAME, self.on_click)

        while True:
            # Read next frame if it's a video and we haven't hit pause
            if is_video and self.is_playing:
                ok, img = video.read()
                if not ok:  # Made it to the end of the video, loop back
                    video = cv2.VideoCapture(self.input)
                    ok, img = video.read()
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            pixel_hsv = hsv[self.pixel[1], self.pixel[0], :]

            # Threshold the HSV image
            h_offset = self.H_max+1 if self.H_min > self.H_max else 0
            hsv_offset = np.mod(hsv[:,:,0] - h_offset, 180) if h_offset > 0 else hsv[:,:,0]  # Hue is mod 180 -> Handle case when threshold goes out of bounds
            mask = cv2.inRange(np.dstack((hsv_offset, hsv[:,:,1:])), (self.H_min-h_offset, self.S_min, self.V_min), (np.mod(self.H_max-h_offset, 180), self.S_max, self.V_max))
            out = cv2.bitwise_and(img, img, mask=mask)

            # Print debugging info if enabled
            if self.show_pixel_info:
                cv2.circle(out, self.pixel, self.PIXEL_INFO_RADIUS, self.PIXEL_INFO_COLOR, self.PIXEL_INFO_THICKNESS)
                cv2.putText(out, "{} - {} ({})".format(self.pixel, pixel_hsv, self.get_str_lims()), (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, self.PIXEL_INFO_COLOR)
            cv2.imshow(self.WIN_NAME, out)

            # Render
            k = cv2.waitKey(1)
            if k == ord(' '):
                self.is_playing = not self.is_playing
            elif k>0:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input: path to an image, a video, a webcam number or an IP camera")
    args = parser.parse_args()

    HSVthreshHelper(args.input).run()
