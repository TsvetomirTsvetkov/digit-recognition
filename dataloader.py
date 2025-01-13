# Standard Library Imports
import array
import numpy as np
import struct

# Constants
TRAINING_LABELS_REL_PATH = "/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
TRAINING_IMAGES_REL_PATH = "/train-images-idx3-ubyte/train-images-idx3-ubyte"
TEST_LABELS_REL_PATH     = "/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
TEST_IMAGES_REL_PATH     = "/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"

MAGIC_LABEL = 2049 
MAGIC_IMAGE = 2051

BYTE_8 = 8

ROW_SIZE = 28
COL_SIZE = 28


class DataLoader:
    def __init__(self, path):
        # Initialize train and test paths
        self.training_labels_filepath = path + TRAINING_LABELS_REL_PATH
        self.training_images_filepath = path + TRAINING_IMAGES_REL_PATH
        self.test_labels_filepath     = path + TEST_LABELS_REL_PATH
        self.test_images_filepath     = path + TEST_IMAGES_REL_PATH

    def _unpack(self, path, expected_magic):
        with open(path, 'rb') as f:
            # Unpack the first 8 bytes
            actual_magic, size = struct.unpack(">II", f.read(BYTE_8))

            # Verify unpacked struct is correct
            if actual_magic != expected_magic:
                raise ValueError("Magic number mismatch")

            data = array.array('B', f.read())

        return data, size

    def _prepare_data(self, labels_path, images_path):
        # Read labels
        labels, size = self._unpack(labels_path, MAGIC_LABEL)
       
        # Read image data
        image_data, _ = self._unpack(images_path, MAGIC_IMAGE)

        # Create helper variables
        images = []
        pixels_per_image = ROW_SIZE * COL_SIZE

        # Prepare images by adding empty arrays
        for i in range(size):
            images.append([0] * pixels_per_image)

        # Get actual image data per image in the 1D array
        for i in range(size):
            img = np.array(image_data[i * pixels_per_image:(i + 1) * pixels_per_image])
            img = img.reshape(ROW_SIZE, COL_SIZE)
            images[i][:] = img

        return images, labels

    def load_data(self):
        # Prepare train data
        x_train, y_train = self._prepare_data(self.training_labels_filepath, self.training_images_filepath)

        # Prepare test data
        x_test, y_test = self._prepare_data(self.test_labels_filepath, self.test_images_filepath)

        return (x_train, y_train), (x_test, y_test)
