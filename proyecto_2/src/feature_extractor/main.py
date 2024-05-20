import cv2
import os
import numpy as np
from tqdm import tqdm  # Optional, for progress bar
from feature_extractor import LocalBinaryPatterns

def extract_lbp(image_path, numPoints=8, radius=1):
  """
  Extracts LBP features from an image and returns the histogram.

  Args:
      image_path (str): Path to the image file.
      numPoints (int, optional): Number of points for LBP calculation. Defaults to 8.
      radius (int, optional): Radius for LBP calculation. Defaults to 1.

  Returns:
      np.ndarray: LBP histogram of the image.
  """
  # Load image (assuming grayscale)
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming OpenCV is installed

  # Extract LBP features and normalize histogram
  lbp_extractor = LocalBinaryPatterns(numPoints, radius)
  features = lbp_extractor.describe(image)
  return features


def process_directory(src_dir, dst_dir, numPoints=8, radius=1):
  """
  Recursively processes all images and masks within a directory structure,
  extracting LBP features and saving them in a corresponding structure.

  Args:
      src_dir (str): Path to the source directory (raw/ in your case).
      dst_dir (str): Path to the destination directory (lbp/ in your case).
      numPoints (int, optional): Number of points for LBP calculation. Defaults to 8.
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
    process_images(os.path.join(src_dir, subdir, "images"),
                   os.path.join(dst_subdir, "images"), numPoints, radius)
    process_images(os.path.join(src_dir, subdir, "masks"),
                   os.path.join(dst_subdir, "masks"), numPoints, radius)
    
def process_images(src_img_dir, dst_img_dir, numPoints=8, radius=1):
  """
  Processes images and masks within a directory, extracting LBP features
  and saving them with the same filename (excluding extension) in the destination directory.

  Args:
      src_img_dir (str): Path to the source directory containing images/masks.
      dst_img_dir (str): Path to the destination directory for LBP features.
      numPoints (int, optional): Number of points for LBP calculation. Defaults to 8.
      radius (int, optional): Radius for LBP calculation. Defaults to 1.
  """
  for filename in tqdm(os.listdir(src_img_dir), desc=f"Processing {src_img_dir}"):
    # Skip non-image files
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
      continue

    # Get image and mask paths
    image_path = os.path.join(src_img_dir, filename)
    base_filename, _ = os.path.splitext(filename)

    # Extract LBP for image
    if os.path.isfile(image_path):
      image_lbp = extract_lbp(image_path, numPoints, radius)
      np.save(os.path.join(dst_img_dir, f"{base_filename}.npy"), image_lbp)

# Set directory containing PNG images
raw_images_dir = "../../data/raw/"
lbp_images_dir = "../../data/lbp/"

process_directory(raw_images_dir, lbp_images_dir)
# Use lbp_features = np.load('path/to/your/file.npy') to read .npy data



