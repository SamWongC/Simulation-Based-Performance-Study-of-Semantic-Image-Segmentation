"""Executable report helpers. Offline validation needs no ROS imports."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def environment(root):
    versions = {}
    for name in ('numpy', 'torch', 'torchvision', 'albumentations', 'opencv-python', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not pip-managed'
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                            capture_output=True, text=True) if __import__('shutil').which('git') else None
    sources = list(Path(root).glob('*.py')) + list((Path(root) / 'ros2_ws/src').rglob('*.py'))
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'versions': versions, 'git_commit': result.stdout.strip() if result and result.returncode == 0 else None,
            'source_sha256': {str(p.relative_to(root)): sha256(p) for p in sources}}


def sample_pairs(root, count=10):
    root = Path(root)
    pairs = [(root / 'images' / f'{i:06d}.png', root / 'labels' / f'{i:06d}.png') for i in range(count)]
    missing = [str(p) for pair in pairs for p in pair if not p.is_file()]
    if missing:
        raise FileNotFoundError('Publish the V1 sample bundle; missing: ' + ', '.join(missing[:4]))
    return pairs


def metric_from_confusion(confusion):
    import numpy as np
    intersection = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.full(len(union), np.nan), where=union > 0)
    return {'miou': float(np.nanmean(iou)) if np.any(union) else None,
            'per_class_iou': [float(x) if np.isfinite(x) else None for x in iou]}


def encode_pair(image_path, label_path):
    import cv2
    import numpy as np
    from carla_labels import CARLA_CITYSCAPES_PALETTE
    bgr = cv2.imread(str(image_path))
    reduced = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or reduced is None or reduced.ndim != 2 or bgr.shape[:2] != reduced.shape:
        raise ValueError(f'Invalid paired arrays: {image_path}, {label_path}')
    representatives = {0: 7, 1: 8, 2: 1, 3: 9, 4: 10, 5: 4, 6: 13, 7: 2, 255: 0}
    if not set(np.unique(reduced)).issubset(representatives):
        raise ValueError('Labels must contain reduced IDs 0-7 or ignore ID 255')
    colours = np.zeros((*reduced.shape, 3), dtype=np.uint8)
    for reduced_id, raw_id in representatives.items():
        colours[reduced == reduced_id] = CARLA_CITYSCAPES_PALETTE[raw_id]
    ok_rgb, rgb = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ok_gt, gt = cv2.imencode('.png', cv2.cvtColor(colours, cv2.COLOR_RGB2BGR))
    if not ok_rgb or not ok_gt:
        raise RuntimeError('Image encoding failed')
    return rgb.tobytes(), gt.tobytes(), reduced


def offline_validate(root, checkpoint, samples, output, device='cpu'):
    import cv2
    import numpy as np
    import torch
    from model import LightUNet
    from dataset import get_transforms
    sys.path.insert(0, str(Path(root) / 'ros2_ws/src/acdcrp_segmentation'))
    from acdcrp_segmentation import utils
    from acdcrp_segmentation.carla_labels import cityscapes_rgb_to_raw_ids, remap_label_image

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Publish the frozen baseline checkpoint: {checkpoint}')
    # Only load checkpoints supplied by the trusted project authors.
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = LightUNet(num_classes=8).to(device)
    model.load_state_dict(state.get('model_state_dict', state))
    model.eval()
    transform = get_transforms(256, 512, augment=False)
    confusion = np.zeros((8, 8), dtype=np.int64)
    rows, timings, manifests = [], [], []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Compare the shared metric with a confusion matrix built by separate code.
    truth = np.array([[0, 0, 1], [1, 2, 255]], dtype=np.uint8)
    prediction = np.array([[0, 1, 1], [1, 2, 7]], dtype=np.uint8)
    inter, union = utils.compute_intersection_union(prediction, truth, 8, 255)
    assert inter.tolist() == [1, 2, 1, 0, 0, 0, 0, 0]
    assert union.tolist() == [2, 3, 1, 0, 0, 0, 0, 0]

    for i, (image_path, label_path) in enumerate(sample_pairs(samples)):
        rgb_bytes, gt_bytes, reduced = encode_pair(image_path, label_path)
        message = SimpleNamespace(data=gt_bytes)
        rgb = utils.decode_compressed_image(SimpleNamespace(data=rgb_bytes))
        target = cv2.resize(reduced, (512, 256), interpolation=cv2.INTER_NEAREST)
        decoded = utils.decode_ground_truth(message)
        np.testing.assert_array_equal(decoded, target)

        def original():
            full = utils.decode_compressed_image(message)
            mapped = remap_label_image(cityscapes_rgb_to_raw_ids(full))
            return cv2.resize(mapped, (512, 256), interpolation=cv2.INTER_NEAREST)

        original()
        utils.decode_ground_truth(message)
        measured = {}
        # Alternate order across frames to reduce systematic ordering bias.
        operations = [('old_ms', original), ('new_ms', lambda: utils.decode_ground_truth(message))]
        for name, operation in operations[::1 if i % 2 == 0 else -1]:
            start = time.perf_counter()
            mask = operation()
            measured[name] = (time.perf_counter() - start) * 1000
            np.testing.assert_array_equal(mask, decoded)
        timings.append({'sample': image_path.name, **measured, 'mismatched_pixels': 0})
        with torch.inference_mode():
            tensor = transform(image=rgb)['image'].unsqueeze(0).to(device)
            pred = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
        valid = target != 255
        confusion += np.bincount(target[valid].astype(np.int64) * 8 + pred[valid], minlength=64).reshape(8, 8)
        row = {'sample': image_path.name, **metric_from_confusion(confusion)}
        rows.append(row)
        manifests.append({'sample': image_path.name, 'image_sha256': sha256(image_path), 'label_sha256': sha256(label_path)})
        if i == 0:
            cv2.imwrite(str(output / 'example_rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output / 'example_prediction.png'), pred)
            cv2.imwrite(str(output / 'example_ground_truth.png'), target)
    np.save(output / 'offline_confusion.npy', confusion)
    result = {'kind': 'fresh_offline_constructed_replay', 'checkpoint_sha256': sha256(checkpoint),
              'epoch': state.get('epoch'), 'device': device, 'samples': manifests, 'steps': rows,
              'decoder_benchmark_source': 'constructed PNG labels, NOT historical live JPEG frames',
              'decoder_trials': timings, 'decoder_old_median_ms': float(np.median([x['old_ms'] for x in timings])),
              'decoder_new_median_ms': float(np.median([x['new_ms'] for x in timings])),
              'metric_unit_test': 'passed', 'environment': environment(root), **metric_from_confusion(confusion)}
    save_json(output / 'offline.json', result)
    return result


def benchmark_captured_live(root, run_dir):
    import cv2
    import numpy as np
    sys.path.insert(0, str(Path(root) / 'ros2_ws/src/acdcrp_segmentation'))
    from acdcrp_segmentation import utils
    from acdcrp_segmentation.carla_labels import cityscapes_rgb_to_raw_ids, remap_label_image
    run_dir = Path(run_dir)
    captured = json.loads((run_dir / 'observation.json').read_text())['captured_gt']
    if len(captured) != 10:
        raise ValueError('Ten distinct live GT messages are required for this benchmark')
    rows = []
    for index, item in enumerate(captured):
        message = SimpleNamespace(data=(run_dir / item['file']).read_bytes())
        def original():
            rgb = utils.decode_compressed_image(message)
            labels = remap_label_image(cityscapes_rgb_to_raw_ids(rgb))
            return cv2.resize(labels, (512, 256), interpolation=cv2.INTER_NEAREST)
        operations = [('old_ms', original), ('new_ms', lambda: utils.decode_ground_truth(message))]
        for _, operation in operations:
            operation()
        row, masks = dict(item), []
        for name, operation in operations[::1 if index % 2 == 0 else -1]:
            start = time.perf_counter()
            masks.append(operation())
            row[name] = 1000 * (time.perf_counter() - start)
        row['mismatched_pixels'] = int(np.count_nonzero(masks[0] != masks[1]))
        rows.append(row)
    save_json(run_dir / 'live_decoder_benchmark.json', rows)
    assert all(row['mismatched_pixels'] == 0 for row in rows), 'Decoder equivalence failed'
    return rows
