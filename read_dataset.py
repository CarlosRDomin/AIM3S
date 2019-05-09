import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from aux_tools import _min, _max, DEFAULT_TIMEZONE, date_range, time_to_float

# NOTE: Dependencies for the protos: pip install --upgrade protobuf grpcio googleapis-common-protos


DEPTH_RESOLUTION = (720, 1280)


def decode_numpy_object(encoded_numpy):
    """Decodes the encoded numpy message"""
    from sensing_proto.sensors_pb2 import DataArray

    DATAARRAY_DECODER = {
        DataArray.INT8: np.int8,
        DataArray.INT16: np.int16,
        DataArray.INT32: np.int32,
        DataArray.INT64: np.int64,
        DataArray.FLOAT32: np.float32,
        DataArray.FLOAT64: np.float64,
    }
    type_ = DATAARRAY_DECODER[encoded_numpy.type]
    array = np.frombuffer(encoded_numpy.data, dtype=type_)
    return array.reshape(encoded_numpy.shape)


def unpack(sensor_data):
    """Unpack the sensor data payload into each timestamp and value"""
    # values = decode_numpy_object(sensor_data.values)[:-1, :].T
    values = decode_numpy_object(sensor_data.values).T
    last_ts = sensor_data.t_latest.ToMilliseconds()/1000.0
    # Compute timestamp for each measurement
    num_readings = values.shape[0]
    offset = num_readings/sensor_data.F_samp
    timestamps = np.linspace(last_ts-offset, last_ts, num_readings)
    return timestamps, values


def parse_weight_calibration(calib_file):
    import json

    if calib_file == "":
        calib_file = "Dataset/weight_calibration.json"

    # Parse the calibration json
    with open(calib_file) as f:
        calib = json.load(f)

    # Generate a dictionary with the ID of each plate as the keys, and {slope, offset} for each ID
    weight_params = {}
    for shelf in calib['shelves']:
        for plate_num, plate in enumerate(shelf['plates']):
            weight_params[plate['id']] = {'slope': plate['slope'], 'offset': plate['offset'], 'shelf_id': shelf['id'], 'plate_num': plate_num+1}  # NOTE: shelf_id and plate_num use 1-based indexing

    return weight_params


def read_weight_data(sensor_folder, weight_calib=None, do_tare=False, is_phidget=False):
    from sensing_proto.sensors_pb2 import SensorData
    if isinstance(weight_calib, str):
        weight_calib = parse_weight_calibration(weight_calib)

    sensor_t = []
    sensor_data = []
    sensor_id = int(sensor_folder.rsplit('_', 1)[1])  # Plate ID is the numbers that come after the '_' on the folder name

    # Parse every file in the folder (order doesn't matter, will sort by timestamp later)
    for filename in os.listdir(sensor_folder):
        with open(os.path.join(sensor_folder, filename), 'rb') as f:
            data = SensorData.FromString(f.read())
            t, weights = unpack(data)
            if weight_calib is not None:
                weights = (weights-weight_calib[sensor_id]['offset'])*weight_calib[sensor_id]['slope']
            sensor_t.append(t)
            sensor_data.append(weights)

    # Stack all the segments together (convert to np.array)
    sensor_t = np.hstack(sensor_t)
    sensor_data = np.vstack(sensor_data) if is_phidget else np.hstack(sensor_data)

    # Segments usually aren't read in (time-series) order -> Sort by timestamp
    t_inds = sensor_t.argsort()
    sensor_t = np.array([DEFAULT_TIMEZONE.localize(datetime.fromtimestamp(t)) for t in sensor_t[t_inds]])
    sensor_data = np.array(sensor_data[t_inds, :] if is_phidget else sensor_data[t_inds])
    if do_tare:
        sensor_data -= sensor_data[0:60].mean().astype(sensor_data.dtype)  # Tare it

    return sensor_t, sensor_data, sensor_id


def read_weights_data(experiment_folder, calib_file="", F_samp=60, *args, **kwargs):
    weight_calib = parse_weight_calibration(calib_file)  # Load calibration file to figure out the plate and shelf arrangement
    shelves = {}  # Keeps track of what shelves have at least 1 plate. Keys are shelf_id's, values are largest plate_id (number of plates in that shelf)
    weights = {}  # We first load all weights and track them by plate_id. Then, we arrange each shelf in a multidimensional numpy array

    # Load all weights
    t_latest_start = datetime.min.replace(tzinfo=DEFAULT_TIMEZONE)
    t_earliest_end = datetime.max.replace(tzinfo=DEFAULT_TIMEZONE)
    for sensor_folder in glob.glob(os.path.join(experiment_folder, "sensors_*")):
        # Read weight
        weight_t, weight_data, plate_id = read_weight_data(sensor_folder, weight_calib, *args, **kwargs)
        # Store results
        weights[plate_id] = {'t': weight_t, 'w': weight_data}
        t_latest_start = _max(weight_t[0], t_latest_start)
        t_earliest_end = _min(weight_t[-1], t_earliest_end)
        # Keep track of how many plates each shelf has
        shelf_id = weight_calib[plate_id]['shelf_id']
        shelves[shelf_id] = _max(weight_calib[plate_id]['plate_num'], shelves.get(shelf_id, 0))

    # Resample the whole fixture as if it had been sampled at fixed F_samp
    t = np.array(list(date_range(t_latest_start, t_earliest_end, timedelta(seconds=1/F_samp))))
    w = np.zeros((len(shelves), max(shelves.values()), len(t)), dtype=np.float32)
    to_float = lambda t_arr: np.array(time_to_float(t_arr, t_latest_start))
    for plate_id, weight in weights.items():
        calib_info = weight_calib[plate_id]
        weight_t = to_float(weight['t'])
        valid_inds = np.hstack((True, np.logical_not(np.equal(weight_t[1:], weight_t[:-1]))))
        w[calib_info['shelf_id']-1, calib_info['plate_num']-1, :] = interp1d(weight_t[valid_inds], weight['w'][valid_inds], kind='cubic', copy=False, assume_sorted=True)(to_float(t))

    return t, w, weights


def read_frame_data(frame_filename):
    from frames_proto.frames_pb2 import Frame

    with open(frame_filename, 'rb') as f:
        data = Frame.FromString(f.read())

        # Parse RGB
        rgb_data = cv2.imdecode(np.frombuffer(data.frame, dtype=np.uint8), -1)

        # Parse depth
        depth_data = np.frombuffer(data.depth, dtype=np.uint16).reshape(DEPTH_RESOLUTION)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.03), cv2.COLORMAP_JET)

        return rgb_data, depth_data, depth_colormap, data


def visualize_weight(base_folder='2019-01-30_PoC/2019-01-30_02:07:40_AIM3S_PoC_rec1/sensors_1'):
    sensor_t, sensor_data = read_weight_data(base_folder)

    # Visualize
    plt.plot(sensor_t, sensor_data)
    plt.show()
    plt.waitforbuttonpress()


def visualize_frame(base_folder='2019-01-30_PoC/2019-01-30_02:07:40_AIM3S_PoC_rec1/frames_26'):
    for filename in os.listdir(base_folder):
        rgb_data, _, depth_colormap, _ = read_frame_data(os.path.join(base_folder, filename))

        # Visualize
        cv2.imshow('RGB', rgb_data)
        cv2.imshow('Depth', depth_colormap)
        cv2.waitKey(100)

        # Only plot first frame for now
        break


if __name__ == '__main__':
    visualize_frame()
    visualize_weight()
