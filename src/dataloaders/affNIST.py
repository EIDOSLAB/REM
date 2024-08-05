import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from glob import glob
try:
    from skimage.util.montage import montage2d
except ImportError as e:
    print('scikit-image is too new',e)
    from skimage.util import montage as montage2d

IMAGE_SIZE_PX = 28
BASE_DIR = '../../data/raw/mnist/MNIST/raw/'
MNIST_FILES = {
    'train': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
}

MNIST_RANGE = {
    'train': (0, 60000)
}

def read_file(file_bytes, header_byte_size, data_size):
  """Discards 4 * header_byte_size of file_bytes and returns data_size bytes."""
  file_bytes.read(4 * header_byte_size)
  return np.frombuffer(file_bytes.read(data_size), dtype=np.uint8)

def read_byte_data(data_dir, split):
  """Extracts images and labels from MNIST binary file.
  Reads the binary image and label files for the given split. Generates a
  tuple of numpy array containing the pairs of label and image.
  The format of the binary files are defined at:
    http://yann.lecun.com/exdb/mnist/
  In summary: header size for image files is 4 * 4 bytes and for label file is
  2 * 4 bytes.
  Args:
    data_dir: String, the directory containing the dataset files.
    split: String, the dataset split to process. It can be one of train, test,
      valid_train, valid_test.
  Returns:
    A list of (image, label). Image is a 28x28 numpy array and label is an int.
  """
  image_file, label_file = (
      os.path.join(data_dir, file_name) for file_name in MNIST_FILES[split])
  start, end = MNIST_RANGE[split]
  with open(image_file, 'r') as f:
    images = read_file(f, 4, end * IMAGE_SIZE_PX * IMAGE_SIZE_PX)
    images = images.reshape(end, IMAGE_SIZE_PX, IMAGE_SIZE_PX)
  with open(label_file, 'r') as f:
    labels = read_file(f, 2, end)

  return zip(images[start:], labels[start:])

def shift_2d(image, shift, max_shift):
  """Shifts the image along each axis by introducing zero.
  Args:
    image: A 2D numpy array to be shifted.
    shift: A tuple indicating the shift along each axis.
    max_shift: The maximum possible shift.
  Returns:
    A 2D numpy array with the same shape of image.
  """
  max_shift += 1
  padded_image = np.pad(image, max_shift, 'constant')
  rolled_image = np.roll(padded_image, shift[0], axis=0)
  rolled_image = np.roll(rolled_image, shift[1], axis=1)
  shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
  return shifted_image


def shift_write_mnist(dataset, filename, shift, pad):
  """Writes the transformed data as tfrecords.
  Pads and shifts the data by adding zeros. Writes each pair of image and label
  as a tf.train.Example in a tfrecords file.
  Args:
    dataset: A list of tuples containing corresponding images and labels.
    filename: String, the name of the resultant tfrecord file.
    shift: Integer, the shift range for images.
    pad: Integer, the number of pixels to be padded
  """
  with tf.python_io.TFRecordWriter(filename) as writer:
    for image, label in dataset:
      padded_image = np.pad(image, pad, 'constant')
      for i in np.arange(-shift, shift + 1):
        for j in np.arange(-shift, shift + 1):
          image_raw = shift_2d(padded_image, (i, j), shift).tostring()
          example = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'height': int64_feature(IMAGE_SIZE_PX + 2 * pad),
                      'width': int64_feature(IMAGE_SIZE_PX + 2 * pad),
                      'depth': int64_feature(1),
                      'label': int64_feature(label),
                      'image_raw': bytes_feature(image_raw),
                  }))
          writer.write(example.SerializeToString())

def read_many_aff(in_paths):
    img_out, label_out = [], []
    for c_path in in_paths:
        a, b = read_affdata(c_path, verbose=False)
        img_out += [a]
        label_out += [b]
    return np.concatenate(img_out, 0), np.concatenate(label_out, 0)
def read_affdata(in_path, verbose=True):
    v = loadmat(in_path)['affNISTdata'][0][0]
    if verbose:
        for k in v:
            print(k.shape)
    img = v[2].reshape((40, 40, -1)).swapaxes(0, 2).swapaxes(1, 2)
    label = v[5][0]
    print(img.shape)
    print(label.shape)
    if verbose:
        plt.imshow(montage2d(img[:81]), cmap='bone')
        plt.show()
    print(label)
    return img, label

class AffNISTDataset(Dataset):
    """AffNIST dataset."""

    def __init__(self, mat_file, transform=None):
        """
        Args:
            mat_file (string): Path to the mat file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images, self.labels = read_affdata(mat_file, False)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        label = torch.from_numpy(np.asarray(self.labels[idx])).long()

        if self.transform:
            image = self.transform(image)

        return image, label