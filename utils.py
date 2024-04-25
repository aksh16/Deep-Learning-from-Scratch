import os
import zipfile
import gzip
import shutil
import numpy as np
import logging
from dataset import Dataset

MNIST_DIR_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "mnist"))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("General Logger")


def extract_dir():
    mnist_zip_loc = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "mnist.zip"))
    if not os.path.isfile(mnist_zip_loc):
        log.error("MNIST zip not in parent of current directory")
    elif os.path.isdir(MNIST_DIR_LOC):
        log.info("MNIST directory exists")
    else:
        try:
            with zipfile.ZipFile(mnist_zip_loc, 'r') as zip_ref:
                zip_ref.extractall(MNIST_DIR_LOC)
        except Exception as e:
            log.error("MNIST zip extraction failed. Probably incomplete zip file", e)
            return
        log.info("MNIST zip extracted")
    return


# Unzips individual files and places in mnist dir
def extract_files():
    if not os.path.isdir(MNIST_DIR_LOC):
        log.error("MNIST zip directory not in parent of current directory")
        return
    zipped_files = [f for f in os.listdir(MNIST_DIR_LOC)]
    if len(zipped_files) > 4:
        log.info("Individual files unzipped")
    unzipped_files = {}
    for f in zipped_files:
        file_name = f.split('.')[0]
        if 'train' in f:
            if 'images' in f:
                unzipped_files['train-imgs'] = file_name
            else:
                unzipped_files['train-labels'] = file_name
        else:
            if 'images' in f:
                unzipped_files['test-imgs'] = file_name
            else:
                unzipped_files['test-labels'] = file_name
        file = (os.path.join(MNIST_DIR_LOC, f))
        unzip_file = (os.path.join(MNIST_DIR_LOC, file_name))
        # Directly extracting the gzip file creates an individual folder for each of the files
        # So create a file and copy the contents there
        try:
            with gzip.open(file, 'rb') as f_in:
                with open(unzip_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception:
            log.error("Individual files zip extraction failed")
    return unzipped_files


# Converts images and labels to numpy arrays
def read_data(files, img_name, label_name):
    img_file = files[img_name]
    img_file = (os.path.join(MNIST_DIR_LOC, img_file))
    try:
        with open(img_file, 'rb') as f:
            metadata = f.read(16)
    except Exception as e:
        log.error("Failed to read image file metadata", e)
        return
    num_imgs = int.from_bytes(metadata[4:8], "big")
    num_rows = int.from_bytes(metadata[8:12], "big")
    num_cols = int.from_bytes(metadata[12:16], "big")
    log.info("Image metadata read")
    img_dim = num_rows * num_cols
    img_stack = []
    try:
        with open(img_file, 'rb') as f:
            f.seek(16)
            imgs = f.read()
    except Exception as e:
        log.error("Failed to read " + img_file, e)
        return
    log.info("Read images from " + img_file)
    for i in range(num_imgs):
        img = [imgs[j] for j in range(i * img_dim, (i + 1) * img_dim)]
        img_flatten = np.array(img)
        img_stack.append(img_flatten)

    imgs = np.array(img_stack)
    label_file = files[label_name]
    label_file = (os.path.join(MNIST_DIR_LOC, label_file))
    try:
        with open(label_file, 'rb') as f:
            f.seek(8)
            labels = f.read()
    except Exception as e:
        log.error("Failed to read " + label_file, e)
        return
    log.info("Read labels from " + label_file)
    label_stack = [labels[i] for i in range(num_imgs)]
    labels = np.array(label_stack)
    print('imgs', type(imgs))
    print('labels', type(labels))
    dataset = Dataset(imgs=imgs, labels=labels, size=num_imgs, dims=(num_rows, num_cols))
    return dataset
