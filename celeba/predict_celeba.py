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
IMG_SIZE = 224
NUM_ATTRIBUTES = 40
ATTRIBUTES_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young']
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"


def process_path(file_path: str) -> Tuple[tf.Tensor, str]:
  """Returns an image and its corresponding path."""
  try:
      raw_img = tf.io.read_file(file_path)
      img = tf.image.decode_png(raw_img, channels=3)
  except:
    print("Could not read the image from %s" % file_path)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [IMG_SIZE + 20, IMG_SIZE + 20])
  img = tf.image.central_crop(img, (IMG_SIZE*1.0)/(IMG_SIZE + 20))
  return img, file_path


def parse_row_train(
    fname: str, attrs: List[float]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns image and attributes for classifier training."""
    raw_img = tf.io.read_file(fname)
    img = tf.image.decode_jpeg(raw_img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE + 20, IMG_SIZE + 20])
    img = tf.image.flip_left_right(img)
    img = tf.image.adjust_brightness(img, -0.1)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.4, 1.0)
    img = tf.image.random_saturation(img, 0, 2)
    img = tf.image.random_jpeg_quality(img, 5, 100)
    rand_stdev = tf.random.normal([1], mean=0., stddev=0.05)
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
    img = img + tf.random.normal([IMG_SIZE, IMG_SIZE, 3], mean=0, stddev=rand_stdev)
    img = tf.clip_by_value(img, 0, 1)
    return img, attrs

def parse_row_test(
    fname: str, attrs: List[float]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns image and attributes for classifier testing."""
    raw_img = tf.io.read_file(fname)
    img = tf.image.decode_jpeg(raw_img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE + 20, IMG_SIZE + 20])
    img = tf.image.central_crop(img, (IMG_SIZE*1.0)/(IMG_SIZE + 20))
    return img, attrs


def get_dataset(image_folder: str) -> tf.data.Dataset:
    imgfile_list = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    datalist = tf.data.Dataset.from_tensor_slices(tf.constant(imgfile_list))
    dataset = datalist.map(process_path).batch(1)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    return dataset


def parse_attr_line(ln):
    """Line parser for the CelebA attribute file line."""
    str_list = ln.split(" ")
    filtered_list = list(filter(lambda x: x !=  "", str_list))
    img_fname = filtered_list[0]
    int_arr = np.array([0 if int(x) == -1 else 1 for x in filtered_list[1:]])
    return img_fname, int_arr


def read_attr_file(
    attr_file: str, img_folder: str) -> Tuple[List[str], List[np.array]]:
    """Reads the attribute file and returns filenames and attributes."""
    ATTR_LIST = []
    FNAME_LIST = []
    with open(attr_file, "r") as fl:
        lines = fl.readlines()[2:]
        for ln in lines:
            name, attrs = parse_attr_line(ln)
            ATTR_LIST.append(attrs)
            FNAME_LIST.append(opj(img_folder, name))
    return FNAME_LIST, ATTR_LIST
    

def get_trainval_datasets(
    img_folder: str,
    attr_file: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Returns train and val datasets for classification."""
    fnames, attr_list = read_attr_file(attr_file, img_folder)
    attr_tensor = tf.data.Dataset.from_tensor_slices(tf.constant(attr_list))
    fname_tensor = tf.data.Dataset.from_tensor_slices(tf.constant(fnames))
    num_train = np.ceil(len(fnames) * 0.8)
    full_dataset = tf.data.Dataset.zip(
        (fname_tensor, attr_tensor))
    train_data = full_dataset.take(num_train).map(parse_row_train).batch(32)
    val_data = full_dataset.skip(num_train).map(parse_row_test).batch(32)
    return train_data, val_data


def get_classifier() -> tf.keras.Model:
    """Returns the classifier model."""
    feature_extractor_layer = hub.KerasLayer(
        MODEL_URL, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dropout(0.2),
        layers.Dense(
            1024, activation=tf.keras.activations.elu, name='hidden_layer_1'),
        layers.Dense(
            512, activation=tf.keras.activations.elu, name='hidden_layer_2'),
        layers.Dense(NUM_ATTRIBUTES, activation='sigmoid', name='output')
    ])
    return model


def maybe_restore_checkpoint_and_compile(
    model: tf.keras.Model, ckpt_path: str) -> tf.keras.Model:
    """Compiles and optionally restores model from checkpoint."""
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=[tf.keras.metrics.binary_accuracy])
    if ckpt_path is not None:
        try:
            model.load_weights(ckpt_path).expect_partial()
            print("Restored checkpoint from %s" % ckpt_path)
        except:
            print("Could not restore checkpoint from %s" % ckpt_path)
    return model


def train_classifier(
    model: tf.keras.Model, ckpt_path: str, 
    train_dset: tf.data.Dataset, val_dset: tf.data.Dataset) -> tf.keras.Model:
    """Trains the attribute classifier."""
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, save_weights_only=True,
        verbose=1, save_freq=100000)
    history = model.fit(
        train_dset,
        epochs=10,
        validation_data=val_dset,
        callbacks=[cp_callback])
    model.save_weights(ckpt_path)
    print("Finished training the classifier and saved to %s" % ckpt_path)
    return model


def predict_attributes(
    model: tf.keras.Model, data: tf.data.Dataset, out_path: str) -> None:
    """Saves the predicted attributes of the translated images to file."""
    with open(out_path, 'w') as out_fl: 
        for img, fname in tqdm.tqdm(data):
            fname = os.path.basename(str(fname.numpy()[0]))[:-1]
            prediction = tf.cast(
                tf.greater(model.predict(img), 0.5), tf.int32).numpy()[0]
            line =' '.join([fname] + [str(x) for x in prediction])
            out_fl.write(line + '\n')
    print("Saved predicted attributes to %s" % out_path)


def main():
    parser = argparse.ArgumentParser(description='Predict attributes for CelebA.')
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
                        help='original CelebA dataset for training.')
    parser.add_argument('--train_attributes', type=str, default=None,
                        help='original CelebA ground truth attributes '
                        'for training.')
    args = parser.parse_args()

    classifier = get_classifier()
    maybe_restore_checkpoint_and_compile(classifier, args.ckpt_dir)

    if args.train_predictors:
        if args.train_data is None:
            raise ValueError('Parameter train_data should be specified if'
            ' train_predictors=True')
        if args.train_attributes is None:
            raise ValueError('Parameter train_attributes should be specified '
            'if train_predictors=True')
        train_data, val_data = get_trainval_datasets(
            args.train_data, args.train_attributes)
        train_classifier(
            classifier,  args.ckpt_dir, train_data, val_data)
    else:
        if args.ckpt_dir is None:
            raise ValueError("Checkpoint directory must be specified.")
    
    transl_data = get_dataset(args.data_dir)
    predict_attributes(classifier, transl_data, args.out_file)

        
if __name__ == "__main__":
    main()