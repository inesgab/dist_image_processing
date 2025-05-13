from utils.mask_utils import calculate_circularity, calculate_centroid, is_duplicate


def filter_valid_masks(
    masks, roi_shape, roi_offset, existing_masks, circularity_threshold=0.85, margin=2
):
    x_min, y_min = roi_offset
    valid_masks = []
    for mask in masks:
        if (
            mask["bbox"][0] < margin
            or mask["bbox"][1] < margin
            or mask["bbox"][0] + mask["bbox"][2] > roi_shape[1] - margin
            or mask["bbox"][1] + mask["bbox"][3] > roi_shape[0] - margin
        ):
            continue
        if calculate_circularity(mask) < circularity_threshold:
            continue
        cx_roi, cy_roi = calculate_centroid(mask["bbox"])
        cx, cy = cx_roi + x_min, cy_roi + y_min
        if any(is_duplicate((cx, cy), (x, y)) for (x, y) in existing_masks):
            print(f"Duplicate mask found at ({cx}, {cy}), skipping.")
            continue
        mask["image_centroid"] = (cx, cy)
        valid_masks.append(mask)
    return valid_masks
