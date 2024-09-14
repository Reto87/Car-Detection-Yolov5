import os
import cv2
import xml.etree.ElementTree as xet
from glob import glob
import warnings
warnings.filterwarnings('ignore')

num_images_to_load = 3000

# Directory paths or could dataset just add appropriate path
label_dir = '/car_dataset-master/train/labels_2/'
image_dir = '/car_dataset-master/train/labels_2/'
yolo_label_dir = '/car_dataset-master/train/labels/'

# make/check dir
os.makedirs(yolo_label_dir, exist_ok=True)

# Load annotations and convert to YOLO format
path = glob(os.path.join(label_dir, '*.xml'))

def convert_to_yolo_format(xmin, xmax, ymin, ymax, img_width, img_height):
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def parse_and_convert(xml_path):
    info = xet.parse(xml_path)
    root = info.getroot()
    filename = root.find('filename').text
    img_path = os.path.join(image_dir, filename)
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape

    for member in root.findall('object'):
        labels_info = member.find('bndbox')
        xmin = int(labels_info.find('xmin').text)
        xmax = int(labels_info.find('xmax').text)
        ymin = int(labels_info.find('ymin').text)
        ymax = int(labels_info.find('ymax').text)

        x_center, y_center, width, height = convert_to_yolo_format(xmin, xmax, ymin, ymax, img_width, img_height)

        class_id = 0 

        yolo_filename = os.path.join(yolo_label_dir, filename.replace('.jpg', '.txt'))
        with open(yolo_filename, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Process all XML files and convert labels to YOLO format
for xml_file in path[:num_images_to_load]:
    parse_and_convert(xml_file)

#Colour print
print("\033[92mLabel conversion to YOLO format completed\033[0m")

#Training
def train_yolo_model():
    train_images = '/car_dataset-master/train/images/'
    val_images = '/car_dataset-master/val/images/'
    yolo_yaml = 'coco.yaml'

    #automatically run traning file in terminal after image convertion and other procedure
    os.system(f'python yolov5/train.py --img 640 --batch 16 --epochs 15 --data {yolo_yaml} --weights yolov5s.pt')

# Run
train_yolo_model()

print("\033[92mYOLOv5 model training completed\033[0m")

def run_inference():
    test_images = '\\car_dataset-master\\test\\images\\'
    #if you cant find yolov5s please check main dir
    os.system(f'python yolov5/detect.py --weights yolov5/runs/train/exp/weights/yolov5s.pt --img 640 --conf 0.25 --source {test_images}')

run_inference()

print("\033[92mYOLOv5 inference completed\033[0m")
