import os
import numpy as np
import PIL
import imageio as imio
import argparse
import tqdm
import glob
from typing import List, Dict, Tuple, Callable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import tensorflow.keras.layers as layers
import tensorflow_hub as hub


opj = os.path.join
IMG_SIZE = 64
ATTRIBUTES = {"identity": 10, "background": 10}


def get_attributes(fname: str) -> Tuple[int, int]:
    """"Parses file name to extract attribute values."""
    bname = os.path.basename(fname.numpy())
    idt = int(bname[:3])
    bg = int(bname[9:12])
    return idt, bg


def get_bname(fname: str) -> str:
    return os.path.basename(fname.numpy())


def parse_fname(fname: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns the image and name for a given filename."""
    raw_img = tf.io.read_file(fname)
    img = tf.image.decode_png(raw_img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    bname = tf.py_function(get_bname, [fname], Tout=tf.string)
    return img, bname


def parse_fname_with_attrs(fname: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns the image and attribute values for a given filename."""
    raw_img = tf.io.read_file(fname)
    img = tf.image.decode_png(raw_img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    attrs = tf.py_function(get_attributes, [fname], Tout=[tf.float32, tf.float32])
    return img, attrs


def get_trainval_datasets(img_folder: str):
    domain_a_files = glob.glob("%s/trainA/*.png" % img_folder)
    domain_b_files = glob.glob("%s/trainB/*.png" % img_folder)
    fname_list = domain_a_files + domain_b_files
    fname_tensor = tf.data.Dataset.from_tensor_slices(tf.constant(fname_list))
    fname_tensor = fname_tensor.shuffle(buffer_size=len(fname_list), seed=123)
    n_train = np.round(len(fname_list) * 0.75)
    train_data = fname_tensor.take(n_train).map(parse_fname_with_attrs)
    val_data = fname_tensor.skip(n_train).map(parse_fname_with_attrs)
    return train_data, val_data


def get_dataset(img_folder: str) -> tf.data.Dataset:
    """Returns an image dataset for attribute prediction."""
    trans_list = glob.glob(os.path.join(img_folder, "*.png"))
    trans_data = tf.data.Dataset.from_tensor_slices(tf.constant(trans_list))
    trans_data = trans_data.map(parse_fname).batch(1)
    return trans_data


def format_line_poses(ln: str) -> Tuple[str, np.array]:
  str_list = ln.split(" ")
  filtered_list = list(filter(lambda x: x != "", str_list))
  int_arr = np.array([int(x) for x in filtered_list[1:]])
  img_fname = filtered_list[0].split("/")[-1]
  return img_fname, int_arr


def get_pose_dict(poses_fpath: str) -> Dict[str, np.array]:
    """"Returns the dictionary with predicted body poses."""
    out_dict = {}
    with open(poses_fpath, "r") as pose_file:
        for ln in pose_file:
            fname, attrs = format_line_poses(ln)
            out_dict[fname] = attrs
    return out_dict


def process_idt(img, attrs):
  return img, tf.reshape(attrs[0], (1, ))
  
  
def process_background(img, attrs):
  return img, tf.reshape(attrs[1], (1, )) 


def get_classifier(num_classes=10):
    """Returns a classifier model"""
    model = tf.keras.Sequential([
        layers.InputLayer([IMG_SIZE, IMG_SIZE, 3]),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model


MODELS_PROCESSORS_TYPE = Tuple[
    List[tf.keras.Model], List[Callable]]


def get_models() -> MODELS_PROCESSORS_TYPE:
    """Returns the classification models for background and identity."""
    idt_model = get_classifier(ATTRIBUTES['identity'])
    bg_model = get_classifier(ATTRIBUTES['background'])
    model_list = [idt_model, bg_model]
    mapping_list = [process_idt, process_background]
    return model_list, mapping_list


def train_model(
    model: tf.keras.Model, 
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    ckpt_path: str, num_epochs: int = 5) -> tf.keras.Model:
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, save_weights_only=True, verbose=1)
    model.fit(
        train_data.batch(32, drop_remainder=True),
        validation_data=val_data.batch(32, drop_remainder=True),
        epochs=num_epochs, verbose=1, callbacks=[cp_callback])
    model.save_weights(ckpt_path)
    return model


def train_all_models(
    models: List[tf.keras.Model],
    processors: List[Callable],
    ckpt_dir: str,
    train_data: tf.data.Dataset, 
    val_data: tf.data.Dataset) -> List[tf.keras.Model]:
    """Trains the attribute predictors."""
    attr_names = list(ATTRIBUTES.keys())
    for i in range(2):
        attr_name = attr_names[i]
        print("Training a model for attribute %s" % attr_name)
        attr_ckpt_path = opj(ckpt_dir, "%s.ckpt" % attr_name)
        train_attr_data = train_data.map(processors[i])
        val_attr_data = val_data.map(processors[i])
        models[i] = train_model(
            models[i],
            train_attr_data,
            val_attr_data,
            attr_ckpt_path,
            num_epochs=5)
    print("Finished training and stored checkpoints at %s" % ckpt_dir)
    return models


def restore_checkpoints(
    models: List[tf.keras.Model], ckpt_dir: str) -> tf.keras.Model:
    """Restores weights from the checkpoint."""
    attr_names = list(ATTRIBUTES.keys())
    for i in range(2):
        attr_name = attr_names[i]
        print("Restoring weights for attribute %s" % attr_name)
        ckpt_path = opj(ckpt_dir, "%s.ckpt" % attr_name)
        try:
            models[i].load_weights(ckpt_path).expect_partial()
        except:
            print("Could not restore weights from %s" % ckpt_path)
    return models


def predict_attributes(
    models: List[tf.keras.Model],
    data: tf.data.Dataset,
    method_poses_fpath: str,
    original_poses_fpath: str,
    out_fpath: str):
    """Stores the predicted attributes for the translated examples."""
    orig_pose_dict = get_pose_dict(original_poses_fpath)
    method_pose_dict = get_pose_dict(method_poses_fpath)

    def predict_single(img: tf.Tensor, fname: tf.Tensor) -> List[int]:
        """Predict attributes for a single example."""
        pred_list = []
        for model in models:
            pred = np.argmax(model(img))
            pred_list.append(pred)

        tr_pose = method_pose_dict[fname]
        cnt_name = "%s.png" % fname.split("_")[0]
        st_name = "%s.png" % fname.split("_")[1]
        cnt_pose = orig_pose_dict[cnt_name]
        st_pose = orig_pose_dict[st_name]
        st_dist = np.sum(abs(tr_pose - st_pose))
        cnt_dist = np.sum(abs(tr_pose - cnt_pose))
        pose_is_close_cnt = int(cnt_dist < st_dist)
        pred_list += [pose_is_close_cnt] 
        return pred_list

    with open(out_fpath, "w") as out_fl:
        for img, fname in tqdm.tqdm(data):
            bname = os.path.basename(fname[0].numpy().decode("ascii"))
            predictions = predict_single(img, bname)
            pred_string = " ".join([bname] + [str(int(x)) for x in predictions] + ['\n'])
            out_fl.write(pred_string)
    print("Finished saving predictions to %s" % out_fpath)


def main():
    parser = argparse.ArgumentParser(description='Predict attributes for Synaction.')
    parser.add_argument('--data_dir',  type=str, default="./translations/",
                        help='directory containing translation results'
                        'named as *content_img_name*_*guidance_img_name*.png')
    parser.add_argument('--out_file', type=str, default="./attr_predictions.txt",
                        help='Output attribute prediction file path')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='folder containing checkpoints of the '
                        'attribute predictors')
    parser.add_argument('--original_poses_file', type=str, default=None,
                        help='file containing pose predictions for the '
                        'original Synaction images')
    parser.add_argument('--translated_poses_file', type=str, default=None,
                        help='file containing pose predictions for the '
                        'translated images')
    parser.add_argument('--train_predictors', type=bool, default=False,
                        help='if True, the predictors will be trained first.')
    parser.add_argument('--train_data', type=str, default=None,
                        help='original Synaction dataset for training.')
    args = parser.parse_args()

    models, processors = get_models()

    if args.train_predictors:
        if args.train_data is None:
            raise ValueError('Parameter train_data should be specified if'
            ' train_predictors=True')
        train_data, val_data = get_trainval_datasets(args.train_data)
        train_all_models(
            models, processors, args.ckpt_dir, train_data, val_data)
    else:
        if args.ckpt_dir is None:
            raise ValueError("Checkpoint directory must be specified.")
        restore_checkpoints(models, args.ckpt_dir)
    
    transl_data = get_dataset(args.data_dir)
    predict_attributes(
        models,
        transl_data,
        args.translated_poses_file,
        args.original_poses_file,
        args.out_file)

        
if __name__ == "__main__":
    main()