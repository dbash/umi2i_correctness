import sys
import glob

import tqdm
import imageio
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

tfl_url = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'

def run_posenet(*inputs):
  tfl_path = tf.keras.utils.get_file('model.tflite', tfl_url, cache_dir='/tmp/')
  model = tf.lite.Interpreter(model_path=tfl_path)
  model.allocate_tensors()

  input_details = model.get_input_details()
  output_details = model.get_output_details()

  assert len(input_details) == len(inputs)

  for input, det in zip(inputs, input_details):
    model.set_tensor(det['index'], input)

  model.invoke()
  outputs = {det['name']: model.get_tensor(det['index']) for det in output_details}
  return outputs


def predict_pose(img):
  assert img.dtype == np.uint8
  assert img.ndim == 3

  orig_size = img.shape[:2]
  final_size = np.max(orig_size)
  xy_borders = [final_size - s for s in orig_size]
  boxed_img = tf.cast(tf.image.resize_with_pad(img, final_size, final_size), tf.uint8)
  pose_boxed = predict_pose_square_img(boxed_img)
  pose_boxed_px = pose_boxed * final_size - np.array(xy_borders)[::-1] / 2
  return pose_boxed_px


def predict_pose_square_img(img):
  assert img.dtype == np.uint8
  input_size, pred_stride, n_joints = 257, 32, 17
  output_size = (input_size - 1) // pred_stride + 1

  img_re = tf.image.resize(img, (input_size, input_size))[..., :3]
  img_re_11 = tf.cast(img_re, 'float32') / 128 - 1
  outputs = run_posenet(img_re_11[None])
  heatmap = outputs['MobilenetV1/heatmap_2/BiasAdd'][0]
  flat_ht = heatmap.reshape(output_size*output_size, n_joints)
  atgmax_idx = np.unravel_index(np.argmax(flat_ht, axis=0), (output_size, output_size))
  argmax_xy = np.vstack(atgmax_idx)[::-1].T
  offsets = outputs['MobilenetV1/offset_2/BiasAdd'][0]
  offests_x = offsets[argmax_xy[:, 1], argmax_xy[:, 0], np.arange(n_joints)]
  offests_y = offsets[argmax_xy[:, 1], argmax_xy[:, 0], n_joints + np.arange(n_joints)]
  offests_xy = np.vstack([offests_x, offests_y])[::-1].T
  positions = offests_xy + argmax_xy * pred_stride
  positions_01 = positions / input_size
  return positions_01


if __name__ == '__main__':
    _, img_folder, out_fn = sys.argv
    
    with open(out_fn, 'w') as f:
        for img_path in tqdm.tqdm(glob.glob(img_folder)):
          img = imageio.imread(img_path)

          pose = predict_pose(img)
          short_idx = [0] + list(range(5, 17))
          short_pose = pose[short_idx]
          coord_str = ['%d' % x for x in np.ravel(short_pose)]
          f.write(img_path + ' ' + ' '.join(coord_str) + '\n')
