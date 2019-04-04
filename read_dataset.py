import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

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
        last_ts = sensor_data.t_latest.ToMilliseconds() / 1000.0
        # Compute timestamp for each measurement
        num_readings = values.shape[0]
        offset = num_readings / sensor_data.F_samp
        timestamps = np.linspace(last_ts - offset, last_ts, num_readings)
        return timestamps, values


def read_weight_data(sensor_folder, do_tare=False, is_phidget=False):
    from sensing_proto.sensors_pb2 import SensorData

    sensor_t = []
    sensor_data = []

    # Parse every file in the folder (order doesn't matter, will sort by timestamp later)
    for filename in os.listdir(sensor_folder):
        with open(os.path.join(sensor_folder, filename), 'rb') as f:
            data = SensorData.FromString(f.read())
            t, weights = unpack(data)
            sensor_t.append(t)
            sensor_data.append(weights)

    # Stack all the segments together (convert to np.array)
    sensor_t = np.hstack(sensor_t)
    sensor_data = np.vstack(sensor_data) if is_phidget else np.hstack(sensor_data)

    # Segments usually aren't read in (time-series) order -> Sort by timestamp
    t_inds = sensor_t.argsort()
    sensor_t = np.array([datetime.fromtimestamp(t) for t in sensor_t[t_inds]])
    sensor_data = np.array(sensor_data[t_inds,:] if is_phidget else sensor_data[t_inds])
    if do_tare:
        sensor_data -= sensor_data[0:60].mean().astype(sensor_data.dtype)  # Tare it

    return sensor_t, sensor_data


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


def visualize_weight():
    base_folder = '2019-01-30_PoC/2019-01-30_02:07:40_AIM3S_PoC_rec1/sensors_1'
    sensor_t, sensor_data = read_weight_data(base_folder)

    # Visualize
    plt.plot(sensor_t, sensor_data)
    plt.show()
    plt.waitforbuttonpress()


def visualize_frame():
    base_folder = '2019-01-30_PoC/2019-01-30_02:07:40_AIM3S_PoC_rec1/frames_26'
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
