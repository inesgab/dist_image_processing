import cv2
import os

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def save_image(image, save_path):
    """Save an image to the specified path."""
    success = cv2.imwrite(save_path, image)
    if not success:
        raise IOError(f"Failed to save image at path: {save_path}")

def get_results_path2(filename):
    """Build the absolute path to the results directory and ensure its existence."""
    # Get the absolute path of the grandparent directory of `src`
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Construct the path to the `results` directory
    results_dir = os.path.join(base_dir, "results")
    # Compose the full path to the target file within `results`
    full_path = os.path.join(results_dir, filename)
    # Create parent directories if they don’t exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    # Return the complete path to the file in `results`
    return full_path

def get_results_path(filename):
    """Build the absolute path to the backup results directory and ensure its existence."""
    # Chemin vers /backup-a/igabert
    backup_dir = os.path.abspath("/backup-a/igabert")
    # Chemin complet vers le fichier dans le sous-dossier results
    results_dir = os.path.join(backup_dir, "results")
    full_path = os.path.join(results_dir, filename)
    # Création du répertoire si nécessaire
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path

def get_data_path(filename):
    """Build the absolute path to a file in the `data` directory."""
    # Get the absolute path of the grandparent directory of `src`
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Construct the path to the `data` directory
    data_dir = os.path.join(base_dir, "data")
    # Return the full path to the file in `data`
    return os.path.join(data_dir, filename)
