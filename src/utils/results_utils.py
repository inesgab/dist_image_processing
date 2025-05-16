import os
import numpy as np
import cv2
from mask_utils import is_duplicate

def calculate_circularity2(mask):
    mask_area = np.sum(mask > 0)
    mask_segmentation = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        print("No contours found in the mask.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        print("Perimeter is zero, cannot calculate circularity.")
        return None

    circularity = (4 * np.pi * mask_area) / (perimeter**2)
    return circularity


def plot_all_masks(base_path, image, t, sort=True):
    """
    Parcourt les dossiers dans base_path, charge les masques et les superpose sur une seule image en utilisant cv2.

    Args:
        base_path (str): Chemin vers le dossier contenant les sous-dossiers.
        image (numpy.ndarray): Image sur laquelle les masques seront superposés.
        t (int): Indice temporel pour sélectionner le masque (mask_t.npy).
    """
    # Créer une copie de l'image pour superposer les masques
    overlay = image.copy()
    margin = 2
    existing_masks = []

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        # Vérifie si le dossier est valide (nommé par un chiffre et des coordonnées)
        if os.path.isdir(folder_path) and "_" in folder_name:
            try:
                # Extraire les coordonnées du nom du dossier
                _, coords = folder_name.split("_")
                x, y = eval(coords)  # Convertit "(50, 60)" en tuple (50, 60)

                # Charger le masque mask.npy
                mask_path = os.path.join(folder_path, f"mask/{t}.npy")
                if os.path.exists(mask_path):
                    mask = np.load(mask_path)
                    mask = (mask * 255).astype(np.uint8)

                    if sort:
                        
                        analysis = cv2.connectedComponents(mask.astype("uint8"))
                        if (
                            analysis[0] > 2
                        ):  # Plus de 1 région connectée (1 pour le fond, 1 pour la région principale)
                            print(f"Mask {(x, y)} contains {analysis[0] - 1} regions, skipping.")
                            continue

                        #area
                        if np.sum(mask)/255 < 9000:
                            print(f"Mask {(x, y)} with area {np.sum(mask)/255} is below threshold, skipping.")
                            continue

                        #circularity
                        if calculate_circularity2(mask) < 0.80:
                            print(
                                f"Mask {(x, y)} with circularity {calculate_circularity2(mask)} is below threshold, skipping."
                            )
                            continue

                        #duplicates
                        if any(is_duplicate((x, y), (ax, ay)) for (ax, ay) in existing_masks):
                            print(f"Duplicate mask found at ({x}, {y}), skipping.")
                            continue
                        existing_masks.append((x, y))

                        
                        
                            
                    full_mask = np.zeros_like(image, dtype=np.uint8)

                    # Redimensionner le masque à sa position sur l'image
                    y_start = max(0, y - mask.shape[0] // 2)
                    y_end = min(image.shape[0], y + mask.shape[0] // 2)
                    x_start = max(0, x - mask.shape[1] // 2)
                    x_end = min(image.shape[1], x + mask.shape[1] // 2)

                    # Calculer les limites du masque original
                    mask_y_start = max(0, mask.shape[0] // 2 - y)
                    mask_y_end = mask_y_start + (y_end - y_start)
                    mask_x_start = max(0, mask.shape[1] // 2 - x)
                    mask_x_end = mask_x_start + (x_end - x_start)

                    # Placer le masque dans le masque de taille complète
                    full_mask[y_start:y_end, x_start:x_end] = mask[
                        mask_y_start:mask_y_end, mask_x_start:mask_x_end
                    ]

                    # Ajouter le masque à l'overlay
                    overlay = cv2.addWeighted(overlay, 1, full_mask, 0.3, 0)
            except Exception as e:
                print(f"Erreur avec le dossier {folder_name}: {e}")

    # Afficher l'image avec les masques superposés
    cv2.imshow("All Masks Superimposed", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Exemple d'utilisation
base_path = "/Users/inesgabert/Documents/LBE/image_processing/results/sbw25-wt-gfp_caa_28c_stock1-01"
image = cv2.imread("/Users/inesgabert/Documents/LBE/image_processing/data/sbw25-wt-gfp_caa_28c_stock1-01/sbw25-wt-gfp_caa_28c_stock1-01_t050_DIC II 40x_ORG.tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plot_all_masks(base_path, image, 20, sort=False)
