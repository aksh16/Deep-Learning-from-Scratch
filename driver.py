from utils import extract_dir, extract_files, read_data
from objects import Vector
import logging

logging.basicConfig(level = logging.INFO)
log = logging.getLogger("General Logger")


def main():
  extract_dir()
  files = extract_files()
  test_set = read_data(files,'test-imgs','test-labels')
  validation_split = 0.4
  train_set = read_data(files,'train-imgs','train-labels')
  train_set.shuffle()
  validation_set = train_set.split(validation_split)
  log.info("Test images with labels")
  test_set.visualize(10)
  log.info("Train images with labels")
  train_set.visualize(10)
  log.info("Validation images with labels")
  validation_set.visualize(10)
  print(type(train_set.imgs))
  x_img = Vector.from_array(train_set.imgs)
  x_label = Vector.from_array(train_set.labels)
  y_img = Vector.from_array(test_set.imgs)
  y_labels = Vector.from_array(test_set.labels)
  print(x_img)

if __name__ == '__main__':
  main()
