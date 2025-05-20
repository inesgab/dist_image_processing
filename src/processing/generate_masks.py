import time
import cv2
import numpy as np
from utils.mask_utils import sort_masks_interactively, crop_image_with_mask, mask_centroid
from utils.file_utils import get_results_path, get_data_path
from utils.image_utils import save_roi_images
from processing.tracking import run_tracking
from processing.filtering_masks import filter_valid_masks


fluo_end_path = "_EGFP_ORG.tif"
dic_end_path = "_DIC II 40x_ORG.tif"

centroid_margin = 25


def get_roi_masks(
    dic_image,
    image_path,
    t1,
    t2,
    roi_coords,
    mask_generator,
    predictor,
    existing_masks,
    circularity_threshold=0.85,
    margin=2,
    msort=False,
    asort=True
):
    """
    Extracts a ROI from a DIC image and generates masks.
    Filters generated masks (circularity, edge proximity, duplicates).
    Valid masks are saved along with their corresponding images at each timepoint.
    Returns the updated list valid masks centroids (for duplicate avoidance due to overlapping).

    Args:
    - dic_image (numpy.ndarray): DIC image from which ROIs are extracted.
    - image_path (str): Base path to access DIC images at different timepoints.
    - t1 (int): First timepoint to process.
    - t2 (int): Last timepoint to process.
    - roi_coords (tuple): ROI coordinates as (x_min, y_min, x_max, y_max).
    - mask_generator (object): Object used to generate masks from the ROI.
    - existing_masks (list): Coordinates of already existing masks to avoid duplicates.
    - circularity_threshold (float, optional): Minimum circularity to consider a mask valid (default 0.85).
    - margin (int, optional): Minimum margin between mask edges and ROI borders (default 2).
    - sort (bool, optional): If True, allows interactive mask sorting (default False).

    Returns:
    - existing_masks (list): Updated list of valid mask coordinates.
    """
    x_min, y_min, x_max, y_max = roi_coords
    roi = dic_image[y_min:y_max, x_min:x_max]
    data_images_path = image_path + "/" + image_path + "_"

    previous_masks_nb = len(existing_masks)

    print(f"Generating masks for ROI at coordinates: {roi_coords}")
    start_time = time.time()
    masks = mask_generator.generate(roi)
    end_time = time.time()
    print(f"Generated {len(masks)} masks in {end_time - start_time:.2f} seconds.")

    valid_masks = filter_valid_masks(
        masks, roi.shape, (x_min, y_min), existing_masks, circularity_threshold, margin, asort=asort
    )
    del masks

    # manually sorting masks
    if msort is True and len(valid_masks) > 0:
        image1 = cv2.imread(
            get_data_path(data_images_path + f"t{t1:03d}" + dic_end_path)
        )
        image2 = cv2.imread(
            get_data_path(data_images_path + f"t{t2:03d}" + dic_end_path)
        )
        roi_t1 = cv2.cvtColor(image1[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
        roi_t2 = cv2.cvtColor(image2[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
        valid_masks = sort_masks_interactively(roi_t1, roi_t2, valid_masks)

    if len(valid_masks) == 0:
        print("No valid mask found.")
        return existing_masks

    print(f"Selected {len(valid_masks)} valid masks.")

    for mask in valid_masks:
        existing_masks.append(mask["image_centroid"])

    # preparing data for tracking
    video_dir = get_data_path(image_path + "/videos")
    save_roi_images(data_images_path, t1, t2, roi_coords, video_dir)
    print(f"Saved ROI images for timepoints {t1} to {t2}.")
    # tracking
    video_segments = run_tracking(predictor, valid_masks, video_dir)

    # saving results
    centroid_list = {}
    output_sizes = {}
    for t in range(1, t2 - t1 + 1):
        t_plot = t + t1 - 1
        fluo_image = cv2.imread(
            get_data_path(data_images_path + f"t{t_plot:03d}" + fluo_end_path)
        )
        dic_image2 = cv2.imread(
            get_data_path(data_images_path + f"t{t_plot:03d}" + dic_end_path)
        )
        fluo_roi = fluo_image[y_min:y_max, x_min:x_max]
        dic_roi = cv2.cvtColor(dic_image2[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
        for idx, mask in video_segments[t].items():
            mask = mask[0]
            # if masks is None or filled with zeros
            if mask is None or np.sum(mask) == 0:
                print(f"Mask {idx} is None, skipping.")
                continue
            if idx not in centroid_list:
                x, y = mask_centroid(mask)
                x_centroid, y_centroid = x + x_min, y + y_min
                centroid_list[idx] = (x_centroid, y_centroid)
            centroid = centroid_list[idx]
            if idx not in output_sizes:
                output_sizes[idx] = None
            mini_image, mini_mask, output_sizes[idx] = crop_image_with_mask(dic_roi, mask, output_sizes[idx])
            mini_fluo, mini_mask , _ = crop_image_with_mask(fluo_roi, mask, output_sizes[idx])
            overlay_t1 = cv2.addWeighted(
                mini_image, 1, (mini_mask > 0).astype(np.uint8) * 255, 0.5, 0
            )
            true_index = previous_masks_nb + idx
            cv2.imwrite(
                get_results_path(image_path + f"/{true_index}_{centroid}/fluo/{t + t1 - 1}.png"),
                mini_fluo,
            )
            cv2.imwrite(
                get_results_path(
                    image_path + f"/{true_index}_{centroid}/overlay/{t + t1 - 1}.png"
                ),
                overlay_t1,
            )
            np.save(
                get_results_path(image_path + f"/{true_index}_{centroid}/mask/{t + t1 - 1}.npy"),
                mini_mask,
            )

    return existing_masks


def save_image_droplets_and_masks(
    folder_path,
    t1,
    t2,
    roi_size,
    mask_generator,
    predictor,
    circularity_threshold=0.85,
    margin=2,
    msort=False,
    asort=True,
    overlap=250
):
    dic_path = get_data_path(
        folder_path + "/" + folder_path + f"_t{t1:03d}" + dic_end_path
    )
    image = cv2.imread(dic_path)
    if image is None:
        print(f"Error: Unable to read image {dic_path}")
        return

    image_height, image_width = image.shape[:2]
    roi_width, roi_height = roi_size
    existing_masks = []
    step = 0
    for y in range(0, image_height, roi_height - overlap):
        for x in range(0, image_width, roi_width - overlap):
            roi_coords = (
                x,
                y,
                min(x + roi_width, image_width),
                min(y + roi_height, image_height),
            )
            print(f"{'-'*10} Treatment {step} {'-'*10}")
            print(
                f"Processing ROI: x_min={roi_coords[0]}, y_min={roi_coords[1]}, x_max={roi_coords[2]}, y_max={roi_coords[3]}"
            )
            existing_masks = get_roi_masks(
                image,
                folder_path,
                t1,
                t2,
                roi_coords,
                mask_generator,
                predictor,
                existing_masks,
                circularity_threshold,
                margin,
                msort,
                asort
            )
            step += 1
