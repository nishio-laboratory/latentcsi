import numpy as np
import torch
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    pipeline,
)
from PIL import Image
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path")
path = Path(parser.parse_args().path)
path = Path("/mnt/nas/esrh/csi_image_data/datasets/walking_test")

photos = [Image.fromarray(i) for i in np.load(path / "photos.npy")]
feature_extractor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
).to(0)
p = pipeline("image-segmentation", "nvidia/segformer-b0-finetuned-ade-512-512")


# ***

seg_maps = []
for i in photos:
    with torch.no_grad():
        inputs = feature_extractor(
            images=Image.fromarray(i), return_tensors="pt"
        ).to(0)
        out = model(**inputs).logits
    seg = torch.argmax(out, dim=1)
    seg_maps.append(seg)

seg_maps = torch.cat(seg_maps)
mapping = {
    0: "wall",
    1: "building",
    2: "sky",
    3: "floor",
    4: "tree",
    5: "ceiling",
    6: "road",
    7: "bed ",
    8: "windowpane",
    9: "grass",
    10: "cabinet",
    11: "sidewalk",
    12: "person",
    13: "earth",
    14: "door",
    15: "table",
    16: "mountain",
    17: "plant",
    18: "curtain",
    19: "chair",
    20: "car",
    21: "water",
    22: "painting",
    23: "sofa",
    24: "shelf",
    25: "house",
    26: "sea",
    27: "mirror",
    28: "rug",
    29: "field",
    30: "armchair",
    31: "seat",
    32: "fence",
    33: "desk",
    34: "rock",
    35: "wardrobe",
    36: "lamp",
    37: "bathtub",
    38: "railing",
    39: "cushion",
    40: "base",
    41: "box",
    42: "column",
    43: "signboard",
    44: "chest of drawers",
    45: "counter",
    46: "sand",
    47: "sink",
    48: "skyscraper",
    49: "fireplace",
    50: "refrigerator",
    51: "grandstand",
    52: "path",
    53: "stairs",
    54: "runway",
    55: "case",
    56: "pool table",
    57: "pillow",
    58: "screen door",
    59: "stairway",
    60: "river",
    61: "bridge",
    62: "bookcase",
    63: "blind",
    64: "coffee table",
    65: "toilet",
    66: "flower",
    67: "book",
    68: "hill",
    69: "bench",
    70: "countertop",
    71: "stove",
    72: "palm",
    73: "kitchen island",
    74: "computer",
    75: "swivel chair",
    76: "boat",
    77: "bar",
    78: "arcade machine",
    79: "hovel",
    80: "bus",
    81: "towel",
    82: "light",
    83: "truck",
    84: "tower",
    85: "chandelier",
    86: "awning",
    87: "streetlight",
    88: "booth",
    89: "television receiver",
    90: "airplane",
    91: "dirt track",
    92: "apparel",
    93: "pole",
    94: "land",
    95: "bannister",
    96: "escalator",
    97: "ottoman",
    98: "bottle",
    99: "buffet",
    100: "poster",
    101: "stage",
    102: "van",
    103: "ship",
    104: "fountain",
    105: "conveyer belt",
    106: "canopy",
    107: "washer",
    108: "plaything",
    109: "swimming pool",
    110: "stool",
    111: "barrel",
    112: "basket",
    113: "waterfall",
    114: "tent",
    115: "bag",
    116: "minibike",
    117: "cradle",
    118: "oven",
    119: "ball",
    120: "food",
    121: "step",
    122: "tank",
    123: "trade name",
    124: "microwave",
    125: "pot",
    126: "animal",
    127: "bicycle",
    128: "lake",
    129: "dishwasher",
    130: "screen",
    131: "blanket",
    132: "sculpture",
    133: "hood",
    134: "sconce",
    135: "vase",
    136: "traffic light",
    137: "tray",
    138: "ashcan",
    139: "fan",
    140: "pier",
    141: "crt screen",
    142: "plate",
    143: "monitor",
    144: "bulletin board",
    145: "shower",
    146: "radiator",
    147: "glass",
    148: "clock",
    149: "flag",
}
ranks = sorted(
    [
        ((seg_maps == i).count_nonzero().item(), mapping[i.item()], i.item())
        for i in seg_maps.unique()
    ],
    reverse=True,
)
print(ranks)

"""
[(1111907, 'wall', 0), (224795, 'floor', 3), (211624, 'person', 12), (27046, 'bench', 69), (18204, 'table', 15), (11194, 'ceiling', 5), (7148, 'chair', 19), (5399, 'base', 40), (4401, 'bed ', 7), (4314, 'counter', 45), (3124, 'bag', 115), (2537, 'television receiver', 89), (1799, 'cabinet', 10), (1469, 'refrigerator', 50), (1015, 'plaything', 108), (762, 'swivel chair', 75), (756, 'computer', 74), (431, 'cradle', 117), (125, 'box', 41), (123, 'conveyer belt', 105), (83, 'monitor', 143), (54, 'basket', 112), (41, 'seat', 31), (25, 'crt screen', 141), (22, 'blanket', 131), (2, 'mirror', 27)]
"""
