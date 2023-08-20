import gzip
import os
import zipfile
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt
import logging
from random import shuffle

logging.basicConfig(level = logging.INFO)
log = logging.getLogger("General Logger")
mnist_dir_location = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"mnist"))

# Unzips mnist zip into current working directory
def extractdataset():
  mnist_zip_location = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"mnist.zip"))
  if not os.path.isfile(mnist_zip_location):
    log.error("MNIST zip not in parent of current directory")
  elif os.path.isdir(mnist_dir_location):
    log.info("MNIST directory exists")
  else:
    try:
      with zipfile.ZipFile(mnist_zip_location, 'r') as zip_ref:
          zip_ref.extractall(mnist_dir_location)
    except Exception as e:
      log.error("MNIST zip extraction failed. Probably incomplete zip file",e)
      return
    log.info("MNIST zip extracted")
  return

#Unzips individual files and places in mnist dir"
def extractfiles():
  if not os.path.isdir(mnist_dir_location):
    log.error("MNIST zip directory not in parent of current directory")
    return
  zipped_files = [f for f in os.listdir(mnist_dir_location)]
  if len(zipped_files) > 4:
    log.info("Individual files unzipped")
    # return
  unzipped_files = {}
  for f in zipped_files:
    if 'train' in f:
      if 'images' in f:
        unzipped_files['train-imgs'] = f.split('.')[0]
      else:
        unzipped_files['train-labels'] = f.split('.')[0]
    elif 't10' in f:
      if 'images' in f:
        unzipped_files['test-imgs'] = f.split('.')[0]
      else:
        unzipped_files['test-labels'] = f.split('.')[0]
  N = len(zipped_files)
  for i in range(N):
    file = (os.path.join(mnist_dir_location,zipped_files[i]))
    unzip_file = (os.path.join(mnist_dir_location,unzipped_files[i]))
    try:
      with gzip.open(file, 'rb') as f_in:
        with open(unzip_file, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)
    except Exception:
      log.error("Individual files zip extraction failed")
      return
  log.info("Files for dataset extracted")
  return unzipped_files

# Converts images and labels to numpy arrays
def readdata(files,img_name,label_name):
  img_file = files[img_name]
  img_file = (os.path.join(mnist_dir_location,img_file))
  try:
    with open(img_file,'rb') as f:
      metadata = f.read(16)
  except Exception as e:
    log.error("Failed to read image file metadata",e)
    return
  num_imgs = int.from_bytes(metadata[4:8],"big")
  num_rows = int.from_bytes(metadata[8:12],"big")
  num_cols = int.from_bytes(metadata[12:16],"big")
  log.info("Image metadata read")
  img_dim = num_rows*num_cols
  img_stack = []
  try:
    with open(img_file,'rb') as f:
      f.seek(16)
      imgs = f.read()
  except Exception as e:
    log.error("Failed to read "+img_file,e)
    return
  log.info("Read images from "+img_file)
  for i in range(num_imgs):
    img = [imgs[j] for j in range(i*img_dim,(i+1)*img_dim)]
    img_flatten = np.array(img)
    img_stack.append(img_flatten)
  
  imgs = np.array(img_stack)
  label_file = files[label_name]
  label_file = (os.path.join(mnist_dir_location,label_file))
  try:
    with open(label_file,'rb') as f:
      f.seek(8)
      labels = f.read()
  except Exception as e:
    log.error("Failed to read "+label_file,e)
    return
  log.info("Read labels from "+label_file)
  label_stack= [labels[i] for i in range(num_imgs)]
  labels = np.array(label_stack)
  dataset = Dataset(imgs=imgs,labels=labels,size=num_imgs,dims=(num_rows,num_cols))
  return dataset

class Dataset(object):
  def __init__(self,imgs:np.array,labels:np.array,size:int,dims:tuple[int,int]) -> None:
    self.imgs = imgs
    self.labels = labels
    self.size = size
    self.dims = dims
  
  def shuffle(self):
    img_label = list(zip(self.labels,self.imgs))
    shuffle(img_label)
    self.imgs = [img[1] for img in img_label]
    self.labels = [img[0] for img in img_label]
    log.info("Shuffled training data")
    return
  
  def visualize(self,num:int):
    for i in range(num):
      img = self.imgs[i].reshape(28,28)
      label = self.labels[i]
      plt.imshow(img, interpolation='nearest')
      log.info("image label:{}".format(label))
      plt.show()
    return
  
  def split(self,percent:float):
    x_imgs = self.imgs[:int(percent*self.size)]
    x_labels = self.labels[:int(percent*self.size)]
    dataset = Dataset(imgs=x_imgs,labels=x_labels,size=int(percent*self.size),dims=self.dims)
    self.imgs = self.imgs[int(percent*self.size):] 
    self.labels = self.labels[int(percent*self.size):]
    self.size = self.size - int(percent*self.size)
    log.info("Shuffled and split training data")
    return dataset

def main():
  extractdataset()
  files = extractfiles()
  test_set = readdata(files,'test-imgs','test-labels')
  validation_split = 0.4
  train_set = readdata(files,'train-imgs','train-labels')
  train_set.shuffle()
  validation_set = train_set.split(validation_split)
  log.info("Test images with labels")
  test_set.visualize(10)
  log.info("Train images with labels")
  train_set.visualize(10)
  log.info("Validation images with labels")
  validation_set.visualize(10)

if __name__ == '__main__':
  main()
