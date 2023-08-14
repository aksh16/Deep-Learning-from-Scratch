import gzip
import os
import zipfile
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging

logging.basicConfig(level = logging.INFO)
log = logging.getLogger("General Logger")

"""Unzips MNIST files into current working directory"""
def extractdataset():
  mnist_zip_location = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"mnist.zip"))
  if not os.path.isfile(mnist_zip_location):
    log.error("MNIST zip not in parent of current directory")
    return
  mnist_dir_location = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"mnist"))
  try:
    with zipfile.ZipFile(mnist_zip_location, 'r') as zip_ref:
        zip_ref.extractall(mnist_dir_location)
  except Exception:
    log.error("MNIST zip extraction failed. Probably incomplete zip file")
    return
  log.info("MNIST zip extracted")
  return

"""Unzips individual files and places them under 'mnist' directory"""
def extractfiles():
  mnist_dir_location = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"mnist"))
  if not os.path.isdir(mnist_dir_location):
    log.error("MNIST zip not in parent of current directory")
    return
  zipped_files = [f for f in os.listdir(mnist_dir_location)]
  unzipped_names = [f.split('.')[0] for f in zipped_files]
  N = len(zipped_files)
  for i in range(N):
    file = (os.path.join(mnist_dir_location,zipped_files[i]))
    unzip_file = (os.path.join(mnist_dir_location,unzipped_names[i]))
    try:
      with gzip.open(file, 'rb') as f_in:
        with open(unzip_file, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)
    except Exception:
      log.error("Individual files zip extraction failed")
      return
  log.info("Files for dataset extracted")
  return

def main():
  extractdataset()
  extractfiles()

if __name__ == '__main__':
  main()
