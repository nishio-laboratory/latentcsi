import cv2
import pickle
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import glob


yolo_dir = Path("/mnt/nas/esrh/csi_image_data/yolo")
if not yolo_dir.exists():
    yolo_dir = Path("/data/yolo")

cfg_path = yolo_dir / "yolov3.cfg"
weights_path = yolo_dir / "yolov3.weights"
names_path = yolo_dir / "coco.names"

# Load class names
with open(names_path, "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Load the network
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = net.getLayerNames()
out_layers = [
    layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()
]


def detect_persons(pil_img, conf_threshold=0.5, nms_threshold=0.4):
    """
    Detect persons in a PIL image using YOLOv3.
    Returns list of bounding boxes: (x, y, w, h)
    """
    img = np.array(pil_img.convert("RGB"))
    img = img[:, :, ::-1]  # RGB to BGR for OpenCV

    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img, 1 / 255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if CLASSES[class_id] == "person" and confidence > conf_threshold:
                center_x, center_y, w, h = (
                    detection[0:4] * np.array([width, height, width, height])
                ).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold
    )
    if len(indices) == 0:
        return []

    result = [[int(j) for j in boxes[i]] for i in indices.flatten()]

    return result


def draw_box_on_pil(
    pil_img, bbox, color=(0, 255, 0), thickness=2, window_name="Image"
):
    """
    Draw a bounding box on a PIL image and show it using OpenCV.

    Args:
        pil_img: PIL.Image instance
        bbox: tuple (x, y, w, h)
        color: rectangle color (B, G, R)
        thickness: thickness of the rectangle
        window_name: window title for cv2.imshow
    """
    # Convert PIL image to OpenCV format (numpy array, BGR color space)
    img_cv = np.array(pil_img.convert("RGB"))[:, :, ::-1]
    img_cv = np.ascontiguousarray(img_cv)

    x, y, w, h = bbox

    # Draw rectangle
    cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, thickness)

    # Show image
    cv2.imshow(window_name, img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path, required=True)
    args = parser.parse_args()

    ckpt_dir = str(args.path).endswith(".ckpt")
    if ckpt_dir:
        photos_paths = glob.glob(str(args.path / "*_p.png"))
    else:
        testsets = list(glob.glob(str(args.path / "testset*")))
        photos_paths = glob.glob(str(Path(testsets[0]) / "*_p.png"))

    bounding_boxes = [
        detect_persons(Image.open(i))
        for i in tqdm(photos_paths, total=len(photos_paths))
    ]

    with open(
        (args.path.parents[0] if ckpt_dir else args.path) / "ts_bboxes.pkl",
        "wb",
    ) as f:
        pickle.dump(bounding_boxes, f)
