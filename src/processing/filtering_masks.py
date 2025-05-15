from utils.mask_utils import calculate_circularity, calculate_centroid, is_duplicate, expand_mask
import cv2


def filter_valid_masks(
    masks, roi_shape, roi_offset, existing_masks, circularity_threshold=0.85, margin=2, asort=True
):
    x_min, y_min = roi_offset
    valid_masks = []
    for mask in masks:
        cx_roi, cy_roi = calculate_centroid(mask["bbox"])
        cx, cy = cx_roi + x_min, cy_roi + y_min
        # margin check
        if asort == False:
            if mask["area"]<62500:
                mask["image_centroid"] = (cx, cy)
                mask['segmentation'] = expand_mask(mask['segmentation'])
                valid_masks.append(mask)
        else:
            if mask["area"]<9000 or mask['area']>62500:
                continue
            
            if (
                mask["bbox"][0] < margin
                or mask["bbox"][1] < margin
                or mask["bbox"][0] + mask["bbox"][2] > roi_shape[1] - margin
                or mask["bbox"][1] + mask["bbox"][3] > roi_shape[0] - margin
            ):
                continue

            # circularity check
            if calculate_circularity(mask) < circularity_threshold:
                print(
                    f"Mask {(cx, cy)} with circularity {calculate_circularity(mask)} is below threshold, skipping."
                )
                continue

            # connected regions check
            analysis = cv2.connectedComponents(mask["segmentation"].astype("uint8"))
            if (
                analysis[0] > 2
            ):  # Plus de 1 région connectée (1 pour le fond, 1 pour la région principale)
                print(f"Mask {(cx, cy)} contains {analysis[0] - 1} regions, skipping.")
                continue

            # duplicates

            if any(is_duplicate((cx, cy), (x, y)) for (x, y) in existing_masks):
                print(f"Duplicate mask found at ({cx}, {cy}), skipping.")
                continue
            mask["image_centroid"] = (cx, cy)
            mask['segmentation'] = expand_mask(mask['segmentation'])
            valid_masks.append(mask)
    return valid_masks
