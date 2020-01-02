import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = "/path/to/data"
META_DIR = "/path/to/metadata"

NUM_POINT = 4096

CLASSES = {
    "ArticTruck": 0,
    "ArticTruckDumptor": 1,
    "ArticTruckLowLoaded": 2,
    "ArticTruckTanker": 3,
    "Bike": 4,
    "Bus": 5,
    "CamperVan": 6,
    "Car": 7,
    "CarWithTrailer": 8,
    "Truck": 9,
    "TruckCarTransporterLoaded": 10,
    "TruckDumptor": 11,
    "TruckLowLoaded": 12,
    "TruckTanker": 13,
    "TruckWithTrailer": 14,
    "Van": 15,
    "VanDelivery": 16,
    "VanPickup": 17,
    "VanPickupWithTrailer": 18,
    "VanWithTrailer": 19,
    "Phantom": 20,
    "ArticVan": 21,
    "TruckCarTransporterEmpty": 22,
    "VanDeliveryWithTrailer": 23
}

metadata_cache = {}


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_asc(filename: str):
    campaign_name, vehicle_name = filename.split("/")
    asc_filename = os.path.join(DATA_DIR, campaign_name, vehicle_name.replace(".vehicle", ".asc"))
    metadata_filename = os.path.join(META_DIR, f"{campaign_name}.csv")

    if campaign_name not in metadata_cache:
        metadata_cache[campaign_name] = pd.read_csv(metadata_filename,
                                                    sep=",", header=None, index_col="fileName",
                                                    names=["fileName", "label", "labelId", "shape",
                                                           "shapeAttributes(1)", "shapeAttributes(2)"])

    class_lbl = metadata_cache[campaign_name].loc[vehicle_name]["label"]
    data_pd = pd.read_csv(asc_filename, sep=" ", header=None, index_col=False, names=["X", "Y", "Z"])

    x = np.array(data_pd["X"].values)
    y = np.array(data_pd["Y"].values)
    z = np.array(data_pd["Z"].values)

    data = np.vstack((x, y, z)).T

    size = len(x)

    if size > NUM_POINT:  # downsample
        print(f"Got size {size} for {filename}")
        rand_idxs = np.random.choice(size, NUM_POINT, replace=False)
        data = data[rand_idxs, :]
    elif size < NUM_POINT:  # upsample
        rand_idxs = np.random.randint(0, size, size=NUM_POINT - size)
        sampled = data[rand_idxs, :]
        data = np.concatenate((data, sampled), axis=0)

    return data, CLASSES[class_lbl]


def loadDataFile(filename):
    with open(filename, "r") as dat_file:
        files = dat_file.read().splitlines()

    data = np.zeros((len(files), NUM_POINT, 3))
    labels = []
    idx = 0
    for f in files:
        d, l = load_asc(f)
        data[idx, :, :] = d
        labels.append(l)
        idx += 1

    return (data, labels)


def load_h5_data_label_seg(h5_filename):
    raise NotImplementedError("Not implementet for this provider")


def loadDataFile_with_seg(filename):
    raise NotImplementedError("Segmentation is not implemented for this provider")
