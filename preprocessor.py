import json
import math
import os
import cv2
import numpy as np
import tensorflow as tf
import albumentations as alb


class ImagePreprocessor:

    augmentation_pipeline = alb.Compose([
        alb.RandomCrop(width=380, height=450),
        alb.HorizontalFlip(p=0.4),
        alb.RandomBrightnessContrast(p=0.3),
        alb.RandomGamma(p=0.3),
        alb.RGBShift(p=0.2),
        alb.VerticalFlip(p=0.5),
        alb.Rotate(limit=40, p=0.3),
        alb.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)],
        bbox_params=alb.BboxParams(format='albumentations',
                                   label_fields=['class_labels']))

    @staticmethod
    def load_image(x):
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img)
        return img

    @staticmethod
    def load_labels(label_path):
        with open(label_path.numpy(), 'r', encoding="utf-8") as f:
            label = json.load(f)

        return [label['class']], label['bbox']

    def get_images(self, path, is_shuffle=False):
        images = tf.data.Dataset.list_files(path, shuffle=is_shuffle)
        images = images.map(self.load_image)
        images = images.map(lambda x: tf.image.resize(x, (224, 224)))
        images = images.map(lambda x: x / 255)
        return images

    def get_labels(self, path, is_shuffle=False):
        labels = tf.data.Dataset.list_files(path, shuffle=is_shuffle)
        labels = labels.map(lambda x: tf.py_function(self.load_labels, [x], [tf.uint8, tf.float16]))
        return labels

    @staticmethod
    def prepare_data(images, labels, batch_size):
        shuffle = math.ceil(len(images) * 1.5)
        data = tf.data.Dataset.zip((images, labels))
        data = data.shuffle(shuffle)
        data = data.batch(batch_size)
        data = data.prefetch(batch_size)
        return data

    @staticmethod
    def resize_images(base_path, folders, destination):
        for partition in folders:
            for file in os.listdir(os.path.join(base_path, partition)):
                if file.endswith('.jpg'):
                    image = file
                    img = cv2.imdecode(np.fromfile(os.path.join('img', partition, image), dtype=np.uint8),
                                       cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        img = cv2.resize(img, (480, 848))
                        cv2.imwrite(os.path.join(destination, partition, image), img)

    def create_and_process_labels(self):
        for folder in ['train', 'test', 'val']:
            for file in os.listdir(os.path.join('resized_images', folder)):
                if not file.endswith('.jpg'):
                    continue
                label_exist = True
                filename = file.split(".")[0]
                input_image_path = os.path.join('resized_images', folder, file)
                input_label_path = os.path.join('resized_images', folder, f'{filename}.json' )
                label = self.load_label(input_label_path)
                if label is None:
                    label_exist = False
                image = self.process_image(input_image_path)
                coordinates = self.process_labels(label, image.shape[1], image.shape[0])
                for i in range(50):
                    augmented = self.perform_augmentation(image, coordinates)
                    output_image_path = os.path.join('augmented_images', folder, f'{filename}{str(i)}.jpg')
                    output_label_path = os.path.join('augmented_images', folder, f'{filename}{str(i)}.json')
                    self.save_image(augmented, output_image_path)
                    self.save_label(file, augmented, output_label_path, label_exist)

    @staticmethod
    def load_label(label_path):
        label = None
        try:
            with open(label_path, 'r') as f:
                label = json.load(f)
        except:
            pass
        return label

    @staticmethod
    def process_labels(label, width, height):
        coordinates = []
        if label:
            coordinates.append(label['shapes'][0]['points'][0][0])
            coordinates.append(label['shapes'][0]['points'][0][1])
            coordinates.append(label['shapes'][0]['points'][1][0])
            coordinates.append(label['shapes'][0]['points'][1][1])
            return list(np.divide(coordinates, [width, height, width, height]))
        return [0, 0, 1, 1]

    @staticmethod
    def process_image(image_path):
        img = None
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (480, 840))
        except Exception as e:
            print(f"Failed to process image file '{image_path}': {e}")
        return img

    def perform_augmentation(self, image, coords):
        augmented = None
        try:
            augmented = self.augmentation_pipeline(image=image, bboxes=[coords], class_labels=['face'])
        except Exception as e:
            print(f"Failed to perform augmentation on image '{image}': {e}")
        return augmented

    @staticmethod
    def save_image(augmented, output_image_path):
        try:
            cv2.imwrite(output_image_path, augmented['image'])
        except Exception as e:
            print(f"Failed to save image file '{output_image_path}': {e}")

    @staticmethod
    def save_label(file, augmented, output_label_path, label_exist):
        annotation = {'image': file}
        try:
            if label_exist and len(augmented['bboxes']) > 0:
                annotation['bbox'] = augmented['bboxes'][0]
                annotation['class'] = 1
            else:
                annotation['bbox'] = [0, 0, 1, 1]
                annotation['class'] = 0
            with open(output_label_path, 'w') as f:
                json.dump(annotation, f)
        except Exception as e:
            print(f"Failed to save label file '{output_label_path}': {e}")
