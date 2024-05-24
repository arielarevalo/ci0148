from pathlib import Path
import cv2
import json
import shutil
import torch
import numpy as np
from skimage import feature

__all__ = ['unpack_images', 'build_label_map', 'load_label_map', 'build_training_data', 'build_lbp', 'build_bf']


class LbpDescriptor:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.num_points, self.radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.num_points + 3),
                                 range=(0, self.num_points + 2))

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist


class BfDescriptor:
    def __init__(self, d, sigma_color, sigma_space):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def describe(self, image):
        filtered_images = cv2.bilateralFilter(image, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        normalized_images = filtered_images / 255.0
        return normalized_images


def unpack_images(image_dir):
    data_path = Path(image_dir)

    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            masks_path = class_dir / 'masks'
            images_sub_path = class_dir / 'images'

            if masks_path.exists():
                shutil.rmtree(masks_path)

            if images_sub_path.exists():
                for file_path in images_sub_path.iterdir():
                    if file_path.is_file():
                        shutil.move(str(file_path), str(class_dir))

                images_sub_path.rmdir()


def build_label_map(image_dir, mapping_file):
    data_path = Path(image_dir)
    class_names = sorted([class_dir.name for class_dir in data_path.iterdir() if class_dir.is_dir()])

    if not class_names:
        raise ValueError(f"No class dirs found in directory: {image_dir}")

    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    with open(mapping_file, 'w') as f:
        json.dump(label_map, f)


def load_label_map(mapping_file):
    with open(mapping_file, 'r') as f:
        label_map = json.load(f)
    return label_map


def build_training_data(image_dir, feature_dir, mapping_file, image_size):
    image_path = Path(image_dir)
    feature_path = Path(feature_dir)
    label_map = load_label_map(mapping_file)

    features = []
    labels = []
    num_classes = len(label_map)

    if num_classes == 0:
        raise ValueError(f"No classes found in mapping file: {mapping_file}")

    for class_name, class_idx in label_map.items():
        class_dir = image_path / class_name
        if class_dir.is_dir():
            class_features = __images_to_features(class_dir, image_size)

            features.extend(class_features)

            # Create one-hot encoded labels for the current class and append them to the labels list
            num_samples = len(class_features)
            label = np.zeros((num_samples, num_classes))
            label[:, class_idx] = 1
            labels.extend(label)

    features = np.array(features)
    labels = np.array(labels)

    feature_path.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(features), feature_path / 'features.pth')
    torch.save(torch.tensor(labels), feature_path / 'labels.pth')


def build_lbp(raw_dir, lbp_dir, num_points=8, radius=1):
    lbp_descriptor = LbpDescriptor(num_points, radius)
    __apply_descriptor(raw_dir, lbp_dir, lbp_descriptor)


def build_bf(raw_dir, bf_dir, d=3, sigma_color=50, sigma_space=50):
    bf_descriptor = BfDescriptor(d, sigma_color, sigma_space)
    __apply_descriptor(raw_dir, bf_dir, bf_descriptor)


def __images_to_features(image_dir, image_size):
    image_path = Path(image_dir)
    vectors = []

    for file_path in image_path.iterdir():
        if file_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            vector = __image_to_feature_vector(file_path, image_size)
            vectors.append(vector)

    if not vectors:
        raise ValueError(f"No images found in directory: {image_dir}")

    return vectors


def __image_to_feature_vector(image_path, image_size):
    image_array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    resized_image_array = cv2.resize(image_array, (image_size, image_size))
    normalized_image_array = resized_image_array / 255.0  # Normalize to [0, 1]
    return normalized_image_array


def __apply_descriptor(raw_dir, output_dir, descriptor):
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    shutil.copy(raw_path / 'labels.pth', output_path / 'labels.pth')

    features = torch.load(raw_path / 'features.pth')
    images = (features.numpy() * 255).astype(np.uint8)

    preprocessed_images = [descriptor.describe(image) for image in images]
    preprocessed_images = np.array(preprocessed_images) / 255.0

    torch.save(torch.tensor(preprocessed_images), output_path / 'features.pth')
