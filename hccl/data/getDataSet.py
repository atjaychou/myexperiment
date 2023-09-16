import os
import xml.etree.ElementTree as ET
from PIL import Image

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    image_path = root.find('path').text

    boxes = []
    labels = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return image_path, boxes, labels

def load_image(image_path):
    image = Image.open(image_path)
    return image

def load_dataset(dataset_dir):
    image_paths = []
    boxes_list = []
    labels_list = []

    annotations_dir = os.path.join(dataset_dir, 'Annotations')
    image_dir = os.path.join(dataset_dir, 'JPEGImages')

    for filename in os.listdir(annotations_dir):
        if filename.endswith('.xml'):
            annotation_path = os.path.join(annotations_dir, filename)
            image_path, boxes, labels = parse_annotation(annotation_path)

            image_paths.append(os.path.join(image_dir, image_path))
            boxes_list.append(boxes)
            labels_list.append(labels)

    return image_paths, boxes_list, labels_list
