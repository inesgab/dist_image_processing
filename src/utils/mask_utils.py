import numpy as np
import cv2


def calculate_circularity(mask):
    """
    Calculate the circularity of a segmented region using mask properties.

    Args:
        mask: Dictionary containing mask properties, including 'segmentation' (binary array) and 'area'.

    Returns:
        Circularity value (float) or None if the mask is invalid.
    """
    area = mask.get("area", None)
    if area is None or area == 0:
        print("Invalid or zero area in mask.")
        return None

    segmentation = mask["segmentation"].astype(np.uint8)
    contours, _ = cv2.findContours(
        segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        print("No contours found in the mask.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        print("Perimeter is zero, cannot calculate circularity.")
        return None

    circularity = (4 * np.pi * area) / (perimeter**2)
    return circularity


def calculate_centroid(bbox):
    """
    Calculate the centroid of a bounding box (bbox).

    Args:
        bbox: List [x_min, y_min, width, height].

    Returns:
        Tuple (x_centroid, y_centroid).
    """
    x_min, y_min, width, height = bbox
    x_centroid = x_min + width / 2
    y_centroid = y_min + height / 2
    return x_centroid, y_centroid


def is_duplicate(centroid1, centroid2, margin=50):
    """
    Check if two masks are duplicates by comparing their centroids.

    Args:


    Returns:
        True if the masks are duplicates, False otherwise.
    """
    distance = np.sqrt(
        (centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2
    )
    return distance <= margin


def sort_masks_interactively(image_t1, image_t2, valid_masks):
    """
    Permet à l'utilisateur de trier les masques validés en affichant chaque masque zoomé.

    Args:
        image1: Image roi à t1.
        image2: Image roi à t2
        valid_masks: Liste des masques validés.

    Returns:
        Liste des masques après tri.
    """
    filtered_masks = []
    print("Appuyez sur 'a' pour accepter ou 'r' pour rejeter le masque.")
    for index, mask in enumerate(valid_masks):
        cropped_image_t1, cropped_mask_t1, _ = crop_image_with_mask(
            image_t1, mask["segmentation"]
        )
        cropped_image_t2, cropped_mask_t2, _ = crop_image_with_mask(
            image_t2, mask["segmentation"]
        )
        if cropped_image_t1 is not None:
            # Superposer les masques sur les images
            overlay_t1 = cv2.addWeighted(
                cropped_image_t1,
                1,
                (cropped_mask_t1 > 0).astype(np.uint8) * 255,
                0.5,
                0,
            )
            overlay_t2 = cv2.addWeighted(
                cropped_image_t2,
                1,
                (cropped_mask_t2 > 0).astype(np.uint8) * 255,
                0.5,
                0,
            )

            # Afficher les deux images côte à côte avec OpenCV
            combined = cv2.hconcat(
                [overlay_t1, overlay_t2]
            )  # Concaténer horizontalement
            cv2.imshow(f"Masque {index}, t1 et t2", combined)

            key = cv2.waitKey(0)  # Attendre une touche
            if key == ord("a"):  # Touche 'a' pour accepter
                filtered_masks.append(mask)
            elif key == ord("r"):  # Touche 'r' pour rejeter
                print(f"Masque {index} rejeté.")
            cv2.destroyWindow(f"Masque {index}, t1 et t2")

    return filtered_masks


def crop_image_with_mask(image, mask, output_size=None, margin_ratio=0.3):
    """
    Crop a fixed-size region centered on the mask centroid, with optional margin.

    Args:
        image: np.ndarray, original image.
        mask: np.ndarray, binary mask.
        output_size: (width, height) tuple or None. If None, use bbox+margin.
        margin_ratio: float, margin around bbox if output_size is None.

    Returns:
        cropped_image, cropped_mask, output_size (if it was None)
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        print("Erreur : Le masque est vide.")
        return None, None, output_size

    # Centroid
    y_c, x_c = int(np.mean(y_indices)), int(np.mean(x_indices))

    if output_size is None:
        # Définir la taille à partir du bbox + marge
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        margin_y = int(height * margin_ratio)
        margin_x = int(width * margin_ratio)
        crop_h = height + 2 * margin_y
        crop_w = width + 2 * margin_x
        # Pour garder carré, prendre le max
        crop_size = max(crop_h, crop_w)
        output_size = (crop_size, crop_size)

    crop_w, crop_h = output_size
    half_w, half_h = crop_w // 2, crop_h // 2

    # Définir les bornes du crop centré
    y_min = max(0, y_c - half_h)
    y_max = min(image.shape[0], y_c + half_h)
    x_min = max(0, x_c - half_w)
    x_max = min(image.shape[1], x_c + half_w)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # Si le crop touche les bords, il peut être plus petit que output_size, donc on pad
    pad_y = output_size[1] - cropped_image.shape[0]
    pad_x = output_size[0] - cropped_image.shape[1]
    if pad_y > 0 or pad_x > 0:
        cropped_image = np.pad(cropped_image, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant') if cropped_image.ndim == 3 else np.pad(cropped_image, ((0, pad_y), (0, pad_x)), mode='constant')
        cropped_mask = np.pad(cropped_mask, ((0, pad_y), (0, pad_x)), mode='constant')

    return cropped_image, cropped_mask, output_size


def mask_centroid(mask):
    """
    Calculate the centroid of a mask.

    Args:
        mask: Binary mask (numpy array).

    Returns:
        Tuple (x_centroid, y_centroid) of the centroid coordinates.
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    x_centroid = int(np.mean(x_indices))
    y_centroid = int(np.mean(y_indices))
    return x_centroid, y_centroid


def expand_mask(mask, pixels=5):
    """
    Agrandit/diffuse un masque binaire en augmentant ses dimensions de `pixels` pixels.

    Args:
        mask (numpy.ndarray): Masque binaire (valeurs 0 ou 1 ou 0-255).
        pixels (int): Nombre de pixels pour agrandir le masque.

    Returns:
        numpy.ndarray: Masque agrandi.
    """

    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)

    # Créer un noyau structurant circulaire
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1)
    )

    # Apply dilatation
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)

    return expanded_mask
