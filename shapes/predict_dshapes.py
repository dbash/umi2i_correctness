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
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers.experimental.preprocessing as preprocessing

opj = os.path.join
IMG_SIZE = (64, 64, 3)
ATTRIBUTES = {
    'floor_hue':10, 'wall_hue':10, 'object_hue':10, 
    'size': 8, 'shape': 4, 'orientation': 15
    }

def read_attr_line(ln) -> Tuple[str, np.array]:
  str_lst = ln.split(' ')
  if "" in str_lst:
    str_lst.remove('')
  img_fname = str_lst[0]
  int_arr = np.array([int(x) for x in str_lst[1:]])
  return img_fname, int_arr


def read_attributes(attr_file: str, data_path: str) -> Dict[str, np.array]:
    """Reads the ground truth attributes from a text file."""
    fname_list, attr_list = [], []
    attr_fs = open(attr_file, 'r')
    with open(attr_file, 'r') as attr_fs:
        for ln in tqdm.tqdm(attr_fs):
            name, attrs = read_attr_line(ln)
            fname = opj(data_path, name)
            fname_list.append(fname)
            attr_list.append(attrs)
    return fname_list, attr_list

def parse_row(fname, attrs):
  """tf.Dataset sample parser for training."""
  raw_img = tf.io.read_file(fname)
  img = tf.image.decode_png(raw_img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [*IMG_SIZE[:-1]])
  return img, attrs


def parse_row_img(fname):
  """tf.Dataset sample parser for prediction."""
  raw_img = tf.io.read_file(fname)
  img = tf.image.decode_png(raw_img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [*IMG_SIZE[:-1]])
  return img, fname



def get_datasets(data_path: str, attr_path: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Returns training and validation splits."""
    fname_list, attr_list = read_attributes(attr_path, data_path)
    
    attr_tensor = tf.data.Dataset.from_tensor_slices(tf.constant(attr_list))
    fname_tensor = tf.data.Dataset.from_tensor_slices(tf.constant(fname_list))

    full_dataset = tf.data.Dataset.zip(
        (fname_tensor, attr_tensor)).shuffle(buffer_size=480000, seed=123)
    train_data = full_dataset.take(400000).map(parse_row)
    val_data = full_dataset.skip(400000).take(80000).map(parse_row)
    return train_data, val_data


DATA_PROCESSOR_TYPE = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def get_data_processors() -> List[DATA_PROCESSOR_TYPE]:
    """Returns a list of data preocessors for each attribute."""
    def process_fhue(img, label):
        return img, tf.cast(tf.reshape(label[0], (1, )), dtype=tf.float32)
  
    def process_whue(img, label):
        return img, tf.cast(tf.reshape(label[1], (1, )), dtype=tf.float32) 

    def process_ohue(img, label):
        return img, tf.cast(tf.reshape(label[2], (1, )), dtype=tf.float32) 

    def process_size(img, label):
        return img, tf.cast(tf.reshape(label[3], (1, )), dtype=tf.float32)

    def process_shape(img, label):
        return img, tf.cast(tf.reshape(label[4], (1, )), dtype=tf.float32) 

    def process_orientation(img, label):
        return img, (tf.cast(label[5], tf.float32) - 7.) / 7.
    
    return [
        process_fhue, process_whue, process_ohue, process_size, 
        process_shape, process_orientation
    ]


def get_classifier(num_classes: int=10) -> tf.keras.Model:
  """Returns a classifier model"""
  model = Sequential([
      layers.InputLayer(IMG_SIZE),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)])
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  return model

def get_regressor() -> tf.keras.Model:
    """Returns a regression model"""
    normalizer = preprocessing.Normalization()
    model = keras.Sequential([
        layers.InputLayer(IMG_SIZE),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Flatten(),
        layers.Dense(1)
    ], name="orientation")
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(1e-5))
    return model

def get_predictors() -> Tuple[List[tf.keras.Model], 
                              List[DATA_PROCESSOR_TYPE]]:
    """Returns 3DShapes attribute prediction models."""
    fhue_model = get_classifier(ATTRIBUTES['floor_hue'])
    whue_model = get_classifier(ATTRIBUTES['wall_hue'])
    ohue_model = get_classifier(ATTRIBUTES['object_hue'])
    size_model = get_classifier(ATTRIBUTES['size'])
    shape_model = get_classifier(ATTRIBUTES['shape'])
    ortn_model = get_regressor()
    model_list = [
        fhue_model, whue_model, ohue_model, size_model, shape_model, ortn_model
        ]
    mapping_list = get_data_processors()
    return model_list, mapping_list


def train_model(model: tf.keras.Model, train_data: tf.data.Dataset, 
    val_data: tf.data.Dataset, ckpt_path: str, num_epochs: int=5):
    """Trains the given model."""
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, save_weights_only=True, verbose=1)
    history = model.fit(
        train_data.batch(32, drop_remainder=True),
        validation_data=val_data.batch(32, drop_remainder=True),
        epochs=num_epochs, verbose=1, callbacks=[cp_callback])
    model.save_weights(ckpt_path)
    return history

def train_predictors(
    predictors: List[tf.keras.Model], processors: List[DATA_PROCESSOR_TYPE],
    train_data: tf.data.Dataset, val_data: tf.data.Dataset, 
    ckpt_path: str=None):
    """Trains all attribute predictors"""
    attr_names = list(ATTRIBUTES.keys())
    if ckpt_path is None:
        ckpt_path = "./checkpoints/"
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
    for i in range(6):
        attr_name = attr_names[i]
        print("Training a model for attribute %s" % attr_name)
        attr_ckpt_path = opj(ckpt_path, "%s.ckpt" % attr_name)
        train_attr_data = train_data.map(processors[i])
        val_attr_data = val_data.map(processors[i])
        _ = train_model(
            predictors[i], train_attr_data, val_attr_data, attr_ckpt_path, num_epochs=1)
    print("Finished training the predictors. Saved all checkpoint in %s" 
          % ckpt_path)


def restore_checkpoints(predictors: List[tf.keras.Model], ckpt_path: str):
    """Restores attribute predictors checkpoints."""
    attr_names = list(ATTRIBUTES.keys())
    for i in range(6):
        attr_name = attr_names[i]
        print("Restoring checkpoint for attribute %s" % attr_name)
        attr_ckpt_path = os.path.join(ckpt_path, "%s.ckpt" % attr_name)
        predictors[i].load_weights(attr_ckpt_path).expect_partial()
    print("Finished loading checkpoints.")


def predict_all_attrs(
    img: tf.Tensor, model_list: List[tf.keras.Model]) -> List[np.int32]:
    """Predicts all attributes for a given image."""
    pred_list = []
    for model in model_list[:-1]:
        pred = np.argmax(model(img))
        pred_list.append(pred)
    pred = np.round(7. * model_list[-1](img) + 7)[0][0]
    pred_list.append(pred)
    return pred_list


def get_translation_dataset(img_dir):
    """Returns a dataset for attribute prediction."""
    trans_list = glob.glob(os.path.join(img_dir, "*.png"))
    print("Translation folder contains %i examples" % len(trans_list))
    fname_list_data = tf.data.Dataset.from_tensor_slices(
        tf.constant(trans_list))
    trans_data = fname_list_data.map(parse_row_img).batch(1)
    return trans_data


def save_attr_predictions_to_file(
    out_fpath: str, trans_data: tf.data.Dataset, models: List[tf.keras.Model]):
    """Saves the attribute predictions to a text file."""
    with open(out_fpath, "w") as out_fl:
        for img, fname in tqdm.tqdm(trans_data):
            bname = os.path.basename(str(fname[0].numpy())[:-1])
            predictions = predict_all_attrs(img, models)
            pred_string = " ".join([bname] + [str(int(x)) for x in predictions] + [str(2), '\n'])
            out_fl.write(pred_string)
    print("Finished saving predictions to %s." % out_fpath)
        

def main():
    parser = argparse.ArgumentParser(description='Predict attributes for 3DShapes.')
    parser.add_argument('--data_dir',  type=str, default="./translations/",
                        help='directory containing translation results'
                        'named as *content_img_name*_*guidance_img_name*.png')
    parser.add_argument('--out_file', type=str, default="./attr_predictions.txt",
                        help='Output attribute prediction file path')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='folder containing checkpoints of the '
                        'attribute predictors')
    parser.add_argument('--train_predictors', type=bool, default=False,
                        help='if True, the predictors will be trained first.')
    parser.add_argument('--train_data', type=str, default=None,
                        help='original 3DShapes dataset for training.')
    parser.add_argument('--train_attributes', type=str, default=None,
                        help='original 3DShapes ground truth attrinutes '
                        'for training.')
    args = parser.parse_args()

    predictors, processors = get_predictors()

    if args.train_predictors:
        if args.train_data is None:
            raise ValueError('Parameter train_data should be specified if'
            ' train_predictors=True')
        if args.train_attributes is None:
            raise ValueError('Parameter train_attributes should be specified '
            'if train_predictors=True')
        train_data, val_data = get_datasets(
            args.train_data, args.train_attributes)
        train_predictors(
            predictors, processors, train_data, val_data, args.ckpt_dir)
    else:
        if args.ckpt_dir is None:
            raise ValueError("Checkpoint directory must be specified.")
        restore_checkpoints(predictors, args.ckpt_dir)
    
    transl_data = get_translation_dataset(args.data_dir)
    save_attr_predictions_to_file(args.out_file, transl_data), predictors)

        
if __name__ == "__main__":
    main()