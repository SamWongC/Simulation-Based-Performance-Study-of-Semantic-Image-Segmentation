"""CARLA semantic labels: the 8-class reduction used in this study, its colour
palette, and helpers to decode CARLA's Cityscapes-coloured label images."""
import numpy as np

REDUCED_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "vegetation",
    4: "vehicle",
    5: "pedestrian",
    6: "sky",
    7: "other",
}

NUM_CLASSES = len(REDUCED_CLASSES)  # 8

# Palette for visualization (RGB) - one colour per reduced class
CLASS_COLORS = {
    0: (128, 64,  128),  # road       - purple
    1: (244, 35,  232),  # sidewalk   - pink
    2: (70,  70,  70),   # building   - dark grey
    3: (107, 142, 35),   # vegetation - olive
    4: (0,   0,   142),  # vehicle    - dark blue
    5: (220, 20,  60),   # pedestrian - red
    6: (70,  130, 180),  # sky        - steel blue
    7: (152, 251, 152),  # other      - pale green
}

# Raw IDs (CARLA 0.9.x):
#   0  Unlabeled        1  Building         2  Fence
#   3  Other            4  Pedestrian       5  Pole
#   6  Road line        7  Road             8  Sidewalk
#   9  Vegetation      10  Vehicles        11  Wall
#  12  Traffic sign    13  Sky             14  Ground
#  15  Bridge          16  Rail track      17  Guard rail
#  18  Traffic light   19  Static          20  Dynamic
#  21  Water           22  Terrain

RAW_TO_REDUCED = {
    0:  255,   # Unlabeled  → unlabeled
    1:  2,   # Building   → building
    2:  7,   # Fence      → other
    3:  7,   # Other      → other
    4:  5,   # Pedestrian → pedestrian
    5:  7,   # Pole       → other
    6:  0,   # Road line  → road
    7:  0,   # Road       → road
    8:  1,   # Sidewalk   → sidewalk
    9:  3,   # Vegetation → vegetation
    10: 4,   # Vehicles   → vehicle
    11: 2,   # Wall       → building
    12: 7,   # Traffic sign → other
    13: 6,   # Sky        → sky
    14: 7,   # Ground     → other
    15: 2,   # Bridge     → building
    16: 7,   # Rail track → other
    17: 7,   # Guard rail → other
    18: 7,   # Traffic light → other
    19: 7,   # Static     → other
    20: 7,   # Dynamic    → other
    21: 7,   # Water      → other
    22: 3,   # Terrain    → vegetation
}

# CARLA 0.9.x CityScapes-style palette: raw_class_id → RGB
# The ROS 2 bridge publishes label images in this palette;
# decoding maps each colour back to its raw ID.
CARLA_CITYSCAPES_PALETTE = {
    0:  (0,   0,   0),
    1:  (70,  70,  70),
    2:  (100, 40,  40),
    3:  (55,  90,  80),
    4:  (220, 20,  60),
    5:  (153, 153, 153),
    6:  (157, 234, 50),
    7:  (128, 64,  128),
    8:  (244, 35,  232),
    9:  (107, 142, 35),
    10: (0,   0,   142),
    11: (102, 102, 156),
    12: (220, 220, 0),
    13: (70,  130, 180),
    14: (81,  0,   81),
    15: (150, 100, 100),
    16: (230, 150, 140),
    17: (180, 165, 180),
    18: (250, 170, 30),
    19: (110, 190, 160),
    20: (170, 120, 50),
    21: (45,  60,  150),
    22: (145, 170, 100),
}

def cityscapes_rgb_to_raw_ids(img_rgb: np.ndarray) -> np.ndarray:
    """Convert a Cityscapes-coloured label image (H×W×3 RGB) to raw CARLA class IDs (H×W).

    Each pixel takes the ID of the nearest palette colour in RGB space, which
    absorbs JPEG compression artefacts in the recorded label stream.
    """
    palette_rgb = np.array(list(CARLA_CITYSCAPES_PALETTE.values()), dtype=np.int32)  # (23, 3)
    palette_ids = np.array(list(CARLA_CITYSCAPES_PALETTE.keys()),  dtype=np.uint8)   # (23,)

    h, w = img_rgb.shape[:2]
    flat = img_rgb.reshape(-1, 3).astype(np.int32)  # (N, 3)

    # Squared L2 distance to each palette colour; looping over the 23 colours avoids an N×23 array
    min_dist = np.full(flat.shape[0], np.iinfo(np.int32).max, dtype=np.int64)
    nearest  = np.zeros(flat.shape[0], dtype=np.uint8)
    for idx, color in enumerate(palette_rgb):
        dist = ((flat - color) ** 2).sum(axis=1)
        better = dist < min_dist
        min_dist[better] = dist[better]
        nearest[better]  = palette_ids[idx]
    return nearest.reshape(h, w)


def remap_label_image(raw_label_array):
    """Map raw CARLA class IDs (H×W) to the 8 reduced classes.

    Unlabeled becomes 255 (ignored in loss and metric); IDs missing from
    RAW_TO_REDUCED default to "other" (7).
    """
    out = np.full_like(raw_label_array, fill_value=7, dtype=np.uint8)  # default: other
    for raw_id, reduced_id in RAW_TO_REDUCED.items():
        out[raw_label_array == raw_id] = reduced_id
    return out

def label_to_color_image(label_array):
    """Colour a reduced label image (H×W) with CLASS_COLORS; ignored pixels (255) stay black."""
    h, w = label_array.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        mask = label_array == class_id
        color_img[mask] = color
    return color_img