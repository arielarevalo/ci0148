from pathlib import Path
import json
import shutil
import numpy as np
import cv2


def unpack_images(data_dir):
    data_path = Path(data_dir)

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


def build_label_map(data_dir, output_file):
    data_path = Path(data_dir)
    class_names = sorted([class_dir.name for class_dir in data_path.iterdir() if class_dir.is_dir()])

    if not class_names:
        raise ValueError(f"No class dirs found in directory: {data_dir}")

    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    with open(output_file, 'w') as f:
        json.dump(label_map, f)


def load_label_map(mapping_file):
    with open(mapping_file, 'r') as f:
        label_map = json.load(f)
    return label_map


def build_training_data(data_dir, mapping_file, output_dir, image_size):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    label_map = load_label_map(mapping_file)

    features = []
    labels = []
    num_classes = len(label_map)

    if num_classes == 0:
        raise ValueError(f"No classes found in mapping file: {mapping_file}")

    for class_name, class_idx in label_map.items():
        class_dir = data_path / class_name
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

    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / 'features.npy', features)
    np.save(output_path / 'labels.npy', labels)


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
