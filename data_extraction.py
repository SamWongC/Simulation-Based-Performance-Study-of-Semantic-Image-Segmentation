"""Extract paired RGB and label images from CARLAB MCAP recordings.

Decodes label frames from the Cityscapes colour palette to the 8 reduced classes
and writes image/label PNG pairs to <output>/images and <output>/labels, e.g.::

    python data_extraction.py --input <recording>.mcap --output data/extracted/V1 --every-nth 10
"""
import argparse
import struct
import sys
from pathlib import Path
 
import cv2
import numpy as np
from mcap_ros2.reader import read_ros2_messages

# Local
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from carla_labels import remap_label_image, cityscapes_rgb_to_raw_ids


def _ros_image_to_numpy(msg):
    """Decode a sensor_msgs/CompressedImage (JPEG or PNG) to an RGB uint8 array (H×W×3).

    CARLAB records the label topic as JPEG (see msg.format), so pixels along class
    boundaries can decode to a neighbouring palette colour. Nearest-colour matching in
    carla_labels.py maps them back to a class; a thin band of boundary-label noise remains.
    """
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(
            f"cv2.imdecode failed — message may be corrupted or empty "
            f"(format={getattr(msg, 'format', 'unknown')})"
        )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # → RGB, R channel at index 0


def extract_mcap(mcap_path: Path, output_dir: Path,
                 rgb_topic: str, label_topic: str,
                 every_nth: int = 1):
    """Read the RGB and label topics of one recording, pair them and save every
    `every_nth` pair as PNGs under `output_dir`."""
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError:
        raise ImportError(
            "mcap_ros2 not found. Install with: pip install mcap-ros2-support"
        )
 
    rgb_buf, label_buf = {}, {}
 
    with open(mcap_path, "rb") as f:
        for msg in read_ros2_messages(f, topics=[rgb_topic, label_topic]):
            t = msg.log_time
            # mcap-ros2-support may return datetime or timedelta instead of int ns
            if hasattr(t, 'timestamp'):       # datetime.datetime
                t = int(t.timestamp() * 1e9)
            elif hasattr(t, 'total_seconds'): # datetime.timedelta
                t = int(t.total_seconds() * 1e9)
            if msg.channel.topic == rgb_topic:
                rgb_buf[t] = msg.ros_msg
            elif msg.channel.topic == label_topic:
                label_buf[t] = msg.ros_msg
 
    _pair_and_save(rgb_buf, label_buf, output_dir, every_nth)


def _pair_and_save(rgb_buf: dict, label_buf: dict,
                   output_dir: Path, every_nth: int):
    """Pair RGB and label frames by arrival order and save every `every_nth` pair
    as PNGs, with labels remapped to the reduced classes."""
    
    img_dir   = output_dir / "images"
    label_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
 
    rgb_times   = sorted(rgb_buf.keys())
    label_times = sorted(label_buf.keys())
 
    if not rgb_times:
        print("WARNING: rgb buffer is empty - check topic names.")
        return
    if not label_times:
        print("WARNING: label buffer is empty - check topic names.")
        return
 
    # Sequential pairing: CARLA publishes RGB and segmentation in the same
    # simulation tick so they arrive in the same order. Timestamp-based matching
    # is fragile because the ROS2 bridge can introduce per-topic delays > 50 ms.
    n = min(len(rgb_times), len(label_times))
    pairs = [(rgb_buf[rgb_times[i]], label_buf[label_times[i]]) for i in range(n)]
    print(f"  Paired {n} frames (rgb={len(rgb_times)}, label={len(label_times)})")
 
    saved = 0
    for i, (rgb_msg, label_msg) in enumerate(pairs):
        if i % every_nth != 0:
            continue
 
        # RGB 
        rgb_np = _ros_image_to_numpy(rgb_msg)
        if rgb_np.ndim == 3:
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        else:
            rgb_bgr = rgb_np
 
        # Label: the ROS2 bridge publishes a CityScapes-coloured image (JPEG).
        # Decode full RGB → nearest CARLA raw class ID, then remap to reduced IDs.
        label_np = _ros_image_to_numpy(label_msg)
        if label_np.ndim == 3:
            raw_ids = cityscapes_rgb_to_raw_ids(label_np)
        else:
            raw_ids = label_np.astype(np.uint8)
 
        reduced = remap_label_image(raw_ids)
 
        fname = f"{saved:06d}.png"
        cv2.imwrite(str(img_dir / fname), rgb_bgr)
        cv2.imwrite(str(label_dir / fname), reduced)
        saved += 1
 
    print(f"Saved {saved} pairs → {output_dir}")

 
def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description="Extract CARLA frames from .bag / .mcap")
    parser.add_argument("--input",        required=False, help="Path to .bag dir or .mcap file")
    parser.add_argument("--output",       required=True,  help="Output directory")
    parser.add_argument("--rgb-topic",    default="/carla/ego_vehicle/rgb/image/compressed")
    parser.add_argument("--label-topic",  default="/carla/ego_vehicle/segmentation/image/compressed")
    parser.add_argument("--every-nth",    type=int, default=1, help="Save every Nth frame")
    parser.add_argument("--dummy",        action="store_true",
                        help="Generate dummy data instead of reading a recording")
    parser.add_argument("--dummy-n",      type=int, default=50)
    args = parser.parse_args()

    out = Path(args.output)

    if args.dummy:
        generate_dummy_dataset(out, n_samples=args.dummy_n)
        return

    if not args.dummy and args.input is None:
        parser.error("--input is required unless --dummy is used")
    inp = Path(args.input)
    if inp.suffix == ".mcap":
        extract_mcap(inp, out, args.rgb_topic, args.label_topic, args.every_nth)
 
if __name__ == "__main__":
    main()