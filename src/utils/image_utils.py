import cv2
import os
from utils.file_utils import get_data_path


def save_roi_images(
    data_images_path: str, t1: int, t2: int, roi_coords: tuple, video_dir: str
) -> None:
    x_min, y_min, x_max, y_max = roi_coords
    os.makedirs(video_dir, exist_ok=True)
    for f in os.listdir(video_dir):
        os.remove(os.path.join(video_dir, f))

    for t in range(t1, t2 + 1):
        img_path = get_data_path(data_images_path + f"t{t:03d}_DIC II 40x_ORG.tif")
        image = cv2.imread(img_path)
        roi = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(
            os.path.join(video_dir, f"{t:03d}.jpg"), roi, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
