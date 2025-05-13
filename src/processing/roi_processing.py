
import cv2
import time
from utils.mask_utils import calculate_circularity, sort_masks_interactively

from utils.file_utils import get_data_path
from processing.fluorescence_processing import calculate_average_fluorescence

fluo_end_path = "_EGFP_ORG.tif"
dic_end_path = "_DIC II 40x_ORG.tif"

def single_roi_fluo(dic_image, image_path1, image_path2, t1, t2, roi_coords, mask_generator, circularity_threshold=0.85, margin=2, tri=False):
    """
    Traite une seule ROI dans une image DIC et calcule la fluorescence moyenne dans cette ROI.

    Args:
        dic_image: Image DIC sous forme de tableau NumPy.
        image_path1: Préfixe du chemin des images de fluorescence.
        image_path2: Suffixe du chemin des images de fluorescence.
        t1: Premier point temporel.
        t2: Dernier point temporel.
        roi_coords: Coordonnées de la ROI (x_min, y_min, x_max, y_max).
        mask_generator: Générateur de masques (par exemple, SamAutomaticMaskGenerator).
        circularity_threshold: Seuil minimal de circularité pour accepter un masque.
        margin: Marge pour considérer qu'un masque est trop proche des bords.

    Returns:
        Dictionnaire contenant les données de fluorescence moyenne pour chaque masque en fonction du temps.
    """
    x_min, y_min, x_max, y_max = roi_coords
    image = dic_image
    # Extraire la ROI
    roi = image[y_min:y_max, x_min:x_max]

    print(f"Generating masks for ROI at coordinates: {roi_coords}")

    # Générer les masques pour la ROI
    start_time = time.time()
    masks = mask_generator.generate(roi)
    end_time = time.time()
    valid_masks = []
    print(f"Generated {len(masks)} masks in {end_time - start_time:.2f} seconds.")

    for mask in masks:
        # Vérifier la non juxtaposition avec les bords de la ROI
        if mask['bbox'][0] < margin or mask['bbox'][1] < margin or \
           mask['bbox'][0] + mask['bbox'][2] > roi.shape[1] - margin or \
           mask['bbox'][1] + mask['bbox'][3] > roi.shape[0] - margin:
            continue

        # Vérifier la circularité
        circularity = calculate_circularity(mask)
        if circularity is None or circularity < circularity_threshold:
            continue

        valid_masks.append(mask)
    del masks
    if tri == True:
        image1 = cv2.imread(get_data_path(image_path1 + f"t0{t1:02d}" + dic_end_path))
        image2 = cv2.imread(get_data_path(image_path1 + f"t0{t2:02d}" + dic_end_path))
        roi_t1 = cv2.cvtColor(image1[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
        roi_t2 = cv2.cvtColor(image2[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
        valid_masks = sort_masks_interactively(roi_t1, roi_t2, valid_masks)

    print(f"Selected {len(valid_masks)} valid masks.")
    

    # Calculer la fluorescence pour chaque point temporel
    fluorescence_data = {}
    for timepoint in range(t1, t2 +1):
        print(f"Processing timepoint {timepoint}...")
        fluo_image_path = get_data_path(image_path1 + f"t0{timepoint:02d}" + fluo_end_path)
        fluo_image = cv2.imread(fluo_image_path)

        fluo_image_roi = fluo_image[y_min:y_max, x_min:x_max]
        for index, mask in enumerate(valid_masks):
            mask_segmentation = mask['segmentation']
            average_fluorescence = calculate_average_fluorescence(fluo_image_roi, mask_segmentation)
            if average_fluorescence is not None:
                if index not in fluorescence_data:
                    fluorescence_data[index] = []
                fluorescence_data[index].append((timepoint, average_fluorescence))
    del valid_masks
    return fluorescence_data


def process_fluo_image(dic_image_path, image_path1, image_path2, t1, t2, roi_size, mask_generator, overlap = 0.1, circularity_threshold=0.85, margin=2, tri=False):
    """
    Parcourt une image entière avec des ROIs, génère des masques pour chaque ROI, calcule la fluorescence moyenne
    pour chaque masque en fonction du temps, puis libère les données des masques avant de passer à la ROI suivante.

    Args:
        image_path: Chemin de l'image DIC.
        t1: Premier point temporel (ex: 75).
        t2: Dernier point temporel (ex: 99).
        roi_size: Taille des ROIs (largeur, hauteur).
        mask_generator: Générateur de masques (par exemple, SamAutomaticMaskGenerator).
        circularity_threshold: Seuil minimal de circularité pour accepter un masque.
        margin: Marge pour considérer qu'un masque est trop proche des bords.

    Returns:
        L'ensemble des fluorescences.
    """
    image = cv2.imread(dic_image_path)
    if image is None:
        print(f"Erreur : Impossible de lire l'image {dic_image_path}")
        return

    image_height, image_width = image.shape[:2]
    roi_width, roi_height = roi_size

    # Dictionnaire pour stocker les fluorescences par masque
    total_fluorescence = []

    # Parcourir l'image avec des ROIs
    for y in range(0, image_height, roi_height):
        for x in range(0, image_width, roi_width):
            roi_coords = (x, y, min(x + roi_width, image_width), min(y + roi_height, image_height))
            print(f"Traitement de la ROI : x_min={roi_coords[0]}, y_min={roi_coords[1]}, x_max={roi_coords[2]}, y_max={roi_coords[3]}")
            fluorescence_data = single_roi_fluo(image, image_path1, image_path2, t1, t2, roi_coords, mask_generator, circularity_threshold, margin, tri)
            if fluorescence_data is not None:
                total_fluorescence.append(fluorescence_data)
            del fluorescence_data

    return total_fluorescence
