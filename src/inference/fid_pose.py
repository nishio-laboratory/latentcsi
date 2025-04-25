import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import (
    AutoProcessor,
    VitPoseForPoseEstimation,
    RTDetrForObjectDetection,
)


def compute_fid_vitpose(images1, images2, device="cuda"):
    """
    Computes the FID between two lists of PIL images using penultimate ViTPose-Plus features
    and torchmetrics' FrechetInceptionDistance with a custom feature extractor.

    Args:
        images1 (List[PIL.Image]): First list of images (e.g., real images)
        images2 (List[PIL.Image]): Second list of images (e.g., generated images)
        device (str): 'cpu' or 'cuda'

    Returns:
        float: FID score
    """

    # Load ViTPose-Plus and human detector models
    processor = AutoProcessor.from_pretrained(
        "usyd-community/vitpose-plus-base"
    )
    vitpose = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-plus-base", device_map=device
    )
    detector_processor = AutoProcessor.from_pretrained(
        "PekingU/rtdetr_r50vd_coco_o365"
    )
    detector = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd_coco_o365", device_map=device
    )

    vitpose.eval().to(device)
    detector.eval().to(device)

    # Define penultimate feature extractor
    class PenultimateViTPosePlus(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

        def forward(self, x):
            vit_out = self.model(x, output_hidden_states=True)
            penultimate = vit_out.hidden_states[-2]
            return penultimate[:, 0, :]

    model = PenultimateViTPosePlus(vitpose).to(device).eval()

    # Human detection and preprocessing
    def detect_humans(images):
        pixel_values = detector_processor(
            images=images, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = detector(**pixel_values)

        results = detector_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor(
                [(img.height, img.width) for img in images]
            ),
            threshold=0.3,
        )

        all_boxes = []
        valid_images = []
        for img, result in zip(images, results):
            person_boxes = result["boxes"][result["labels"] == 0]
            if len(person_boxes) == 0:
                continue  # skip image if no humans detected
            boxes = person_boxes.cpu().numpy()
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # x2 - x1 = width
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # y2 - y1 = height
            all_boxes.append(boxes.tolist())
            valid_images.append(img)
        return valid_images, all_boxes

    def preprocess_images(images):
        valid_images, boxes = detect_humans(images)
        if not valid_images:
            return None
        inputs = processor(
            images=valid_images, boxes=boxes, return_tensors="pt"
        )
        return inputs["pixel_values"]

    imgs1_tensor = preprocess_images(images1)
    imgs2_tensor = preprocess_images(images2)

    if imgs1_tensor is None or imgs2_tensor is None:
        raise ValueError("No valid human detections in one of the image sets.")

    imgs1_tensor = imgs1_tensor.to(device)
    imgs2_tensor = imgs2_tensor.to(device)

    # Initialize FID metric with custom feature extractor
    fid_metric = FrechetInceptionDistance(
        feature=model, normalize=True, reset_real_features=False
    ).to(device)

    fid_metric.update(imgs1_tensor, real=True)
    fid_metric.update(imgs2_tensor, real=False)

    fid = fid_metric.compute().item()
    return fid
