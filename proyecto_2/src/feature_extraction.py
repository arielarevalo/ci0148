import cv2
import os
import numpy as np
from skimage import feature
from tqdm import tqdm  # Optional, for progress bar


class LocalBinaryPatterns:
    def __init__(self, num_points, radius):
        # store the number of points and radius
        self.numPoints = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


def lbp(src_dir, dst_dir, num_points=8, radius=1):
    """
    Recursively processes all images and masks within a directory structure,
    extracting LBP features and saving them in a corresponding structure.

    Args:
        src_dir (str): Path to the source directory (raw/ in your case).
        dst_dir (str): Path to the destination directory (lbp/ in your case).
        num_points (int, optional): Number of points for LBP calculation. Defaults to 8.
        radius (int, optional): Radius for LBP calculation. Defaults to 1.
    """
    for subdir in os.listdir(src_dir):
        # Skip non-directory entries
        if not os.path.isdir(os.path.join(src_dir, subdir)):
            continue

        # Create corresponding subdirectory in destination
        dst_subdir = os.path.join(dst_dir, subdir)
        os.makedirs(dst_subdir, exist_ok=True)  # Create directory if it doesn't exist

        # Process images and masks within the subdirectory
        _process_images(os.path.join(src_dir, subdir, "images"),
                        os.path.join(dst_subdir, "images"), num_points, radius)
        # _process_images(os.path.join(src_dir, subdir, "masks"),
        #                 os.path.join(dst_subdir, "masks"), num_points, radius)


def _extract_lbp(image_path, num_points=8, radius=1):
    """
    Extracts LBP features from an image and returns the histogram.

    Args:
        image_path (str): Path to the image file.
        num_points (int, optional): Number of points for LBP calculation. Defaults to 8.
        radius (int, optional): Radius for LBP calculation. Defaults to 1.

    Returns:
        np.ndarray: LBP histogram of the image.
    """
    # Load image (assuming grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming OpenCV is installed

    # Extract LBP features and normalize histogram
    lbp_extractor = LocalBinaryPatterns(num_points, radius)
    features = lbp_extractor.describe(image)
    return features

def _process_images(src_img_dir, dst_img_dir, num_points=8, radius=1):
    """
    Processes images and masks within a directory, extracting LBP features
    and saving them with the same filename (excluding extension) in the destination directory.

    Args:
        src_img_dir (str): Path to the source directory containing images/masks.
        dst_img_dir (str): Path to the destination directory for LBP features.
        num_points (int, optional): Number of points for LBP calculation. Defaults to 8.
        radius (int, optional): Radius for LBP calculation. Defaults to 1.
    """
    for filename in tqdm(os.listdir(src_img_dir), desc=f"Processing {src_img_dir}"):
        # Skip non-image files
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Get image and mask paths
        image_path = os.path.join(src_img_dir, filename)
        base_filename, _ = os.path.splitext(filename)
        os.makedirs(dst_img_dir, exist_ok=True)

        # Extract LBP for image
        if os.path.isfile(image_path):
            image_lbp = _extract_lbp(image_path, num_points, radius)
            np.save(os.path.join(dst_img_dir, f"{base_filename}.npy"), image_lbp)

def bf(src_dir, dst_dir, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Recursively processes all images and masks within a directory structure,
    performing Bilateal Filtering and saving them in a corresponding structure.

    Args:
        src_dir (str): Path to the source directory (raw/ in your case).
        dst_dir (str): Path to the destination directory (bf/ in your case).
        d (int, optional): Diameter of each pixel neighborhood.
        sigmaColor (int, optional): Value of \sigma in the color space.
            The greater the value, the colors farther to each other will start to get mixed.
        sigmaSpace (int, optional): Value of \sigma in the coordinate space.
            The greater its value, the more further pixels will mix together
            , given that their colors lie within the sigmaColor range.        
    """
    for subdir in os.listdir(src_dir):
        # Skip non-directory entries
        if not os.path.isdir(os.path.join(src_dir, subdir)):
            continue

        # Create corresponding subdirectory in destination
        dst_subdir = os.path.join(dst_dir, subdir)
        # Create directory if it doesn't exist
        os.makedirs(dst_subdir, exist_ok=True)
        # Process images and masks within the subdirectory
        _filter_images(os.path.join(src_dir, subdir, "images"),
                        os.path.join(dst_subdir, "images"), d, sigmaColor, sigmaSpace)
        # _filter_images(os.path.join(src_dir, subdir, "masks"),
        #                 os.path.join(dst_subdir, "masks"), d, sigmaColor, sigmaSpace)

def _filter_image(image_path, d=9, sigmaColor=25, sigmaSpace=25):
    """
    Performs bilateral filtering on an image and returns the filtered image.

    Args:
        image_path (str): Path to the image file.
        d (int, optional): Diameter of each pixel neighborhood.
        sigmaColor (int, optional): Value of \sigma in the color space.
            The greater the value, the colors farther to each other will start to get mixed.
        sigmaSpace (int, optional): Value of \sigma in the coordinate space.
            The greater its value, the more further pixels will mix together
            , given that their colors lie within the sigmaColor range.
    Returns
        cv2.image
    """
    # Load image (assuming grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    filtered_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return filtered_image

def _filter_images(src_img_dir, dst_img_dir, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Processes images and masks within a directory, performing bilaterl filtering
    and saving them with the same filename (excluding extension) in the destination directory.

    Args:
        src_img_dir (str): Path to the source directory containing images/masks.
        dst_img_dir (str): Path to the destination directory for Bilaterl Filtering.
        d (int, optional): Diameter of each pixel neighborhood.
        sigmaColor (int, optional): Value of \sigma in the color space.
            The greater the value, the colors farther to each other will start to get mixed.
        sigmaSpace (int, optional): Value of \sigma in the coordinate space.
            The greater its value, the more further pixels will mix together
            , given that their colors lie within the sigmaColor range.

    """
    for filename in tqdm(os.listdir(src_img_dir), desc=f"Processing {src_img_dir}"):
        # Skip non-image files
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Get image and mask paths
        image_path = os.path.join(src_img_dir, filename)
        base_filename, _ = os.path.splitext(filename)
        os.makedirs(dst_img_dir, exist_ok=True)

        # Extract LBP for image
        if os.path.isfile(image_path):
            filtered_image = _filter_image(image_path, d, sigmaColor, sigmaSpace)
            result = cv2.imwrite(os.path.join(dst_img_dir, f"{base_filename}.png"), filtered_image)
