import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--data_file', type=str, default="data/modelnet40_ply_hdf5_2048/ply_data_test0.h5")
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()

BATCH_SIZE = 1
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model)  # import network module
DUMP_DIR = FLAGS.dump_dir
DATA_FILE = FLAGS.data_file

if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)

LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate_one.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
               open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]

HOSTNAME = socket.gethostname()

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    is_training = False

    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points, global_features = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'global_features': global_features}

    eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False

    current_data, current_label = provider.loadDataFile(DATA_FILE)
    size = 512
    current_data = current_data[:, 0:size, :]
    rand_idxs = np.random.randint(0, size, size=NUM_POINT - size)
    sampled = current_data[:, rand_idxs, :]
    current_data = np.concatenate((current_data, sampled), axis=1)
    current_label = np.squeeze(current_label)
    print(current_data.shape)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx

        assert cur_batch_size == 1

        # assume first point is not very influencial
        non_influencial_point = current_data[start_idx, 0, :].copy()

        glob_features = []
        for point in range(NUM_POINT):
            # set all points = non_influencial_point except point
            one_hot_data = current_data[start_idx:end_idx, :, :].copy()
            idxs = [x for x in np.arange(0, NUM_POINT) if x != point]
            one_hot_data[0, idxs] = non_influencial_point

            # Calculate and store feature vector (after max pooling)
            feed_dict = {ops['pointclouds_pl']: one_hot_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}

            global_features_val = sess.run([ops['global_features']], feed_dict=feed_dict)
            glob_features.append(global_features_val)

        # Find the most influential point for each element in the global feature vector
        max_feature_value = np.zeros(len(glob_features[0])) - 10
        max_feature_idxs = np.zeros(len(glob_features[0]), dtype=int)
        for point_idx in range(len(glob_features)):
            point_feature_value = glob_features[point_idx]
            max_feature_idxs = np.where(point_feature_value > max_feature_value, point_idx, max_feature_idxs)
            max_feature_value = np.where(point_feature_value > max_feature_value, point_feature_value,
                                         max_feature_value)

        critical_points = current_data[start_idx:end_idx, max_feature_idxs, :][0, 0, :][0]
        pc_frame = pd.DataFrame(data=critical_points, index=None, columns=['X', 'Y', 'Z'])
        pc_frame.to_csv(os.path.join(DUMP_DIR, f"{start_idx}.asc"), sep=" ", header=False, index=False)

        pc_frame = pd.DataFrame(data=current_data[start_idx, :, :], index=None, columns=['X', 'Y', 'Z'])
        pc_frame.to_csv(os.path.join(DUMP_DIR, f"{start_idx}_orig.asc"), sep=" ", header=False, index=False)

        print(f"Processed {start_idx}.asc")
        # Aggregating END



if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
