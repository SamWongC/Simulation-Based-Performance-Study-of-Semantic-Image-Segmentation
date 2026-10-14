"""Bounded ROS 2 report experiment; run after sourcing Jazzy and the workspace."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from report_repro import encode_pair, sample_pairs, save_json, sha256


def stamp(message):
    return [message.header.stamp.sec, message.header.stamp.nanosec]


def worker(role, output, ros_args):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from acdcrp_segmentation.inference_node import SegmentationInferenceNode
    from acdcrp_segmentation.evaluation_node import SegmentationEvaluationNode
    from acdcrp_segmentation.utils import compute_iou_from_counts
    rclpy.init(args=ros_args)
    stream = (output / 'pairs.jsonl').open('w', encoding='utf-8') if role == 'evaluation' else None

    class RecordedEvaluation(SegmentationEvaluationNode):
        def synchronized_callback(self, prediction, truth):
            count = self.pair_count
            super().synchronized_callback(prediction, truth)
            if self.pair_count != count:
                _, miou = compute_iou_from_counts(self.total_intersection, self.total_union)
                stream.write(json.dumps({'pair': self.pair_count, 'prediction_stamp': stamp(prediction),
                                         'truth_stamp': stamp(truth), 'miou': float(miou),
                                         'wall_monotonic': time.monotonic()}) + '\n')
                stream.flush()

    node = None
    try:
        node = SegmentationInferenceNode() if role == 'inference' else RecordedEvaluation()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if role == 'evaluation':
                save_json(output / 'evaluation_final.json', {
                    'pair_count': node.pair_count, 'intersection': node.total_intersection.tolist(),
                    'union': node.total_union.tolist()})
            else:
                save_json(output / 'inference_final.json', {'frame_count': node.frame_count, 'device': str(node.device)})
            node.destroy_node()
        if stream:
            stream.close()
        if rclpy.ok():
            rclpy.shutdown()


def stop_process(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def run(args):
    import rclpy
    import numpy as np
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import Float32
    from rclpy.qos import qos_profile_sensor_data as qos
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.mode == 'replay':
        pairs = sample_pairs(args.samples)
    prefix = '/report_' + uuid.uuid4().hex[:10]
    rgb_topic = prefix + '/rgb' if args.mode == 'replay' else args.rgb_topic
    gt_topic = prefix + '/truth' if args.mode == 'replay' else args.gt_topic
    prediction_topic, metric_topic = prefix + '/mask', prefix + '/miou'
    configuration = {'mode': args.mode, 'seconds': args.seconds, 'checkpoint_sha256': sha256(checkpoint),
                     'rgb_topic': rgb_topic, 'gt_topic': gt_topic, 'namespace_prefix': prefix,
                     'middleware': {k: os.environ.get(k) for k in ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'ZENOH_CONFIG_OVERRIDE')},
                     'device_requested': args.device, 'sync_queue': 20, 'sync_slop': 0.15}
    save_json(output / 'configuration.json', configuration)
    processes, logs = [], []
    node = None
    success = False
    measurements = {'rgb': [], 'gt': [], 'prediction': [], 'metric': []}
    captured, observed_masks, observed_metrics = [], {}, []
    subscriptions = []
    started = None
    try:
        rclpy.init(args=[])
        node = rclpy.create_node('report_observer_' + uuid.uuid4().hex[:8])
        bridge = CvBridge()
        def receive(name, msg):
            if started is not None:
                measurements[name].append(time.monotonic())
            if name == 'prediction' and args.mode == 'replay':
                observed_masks[tuple(stamp(msg))] = np.asarray(bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')).copy()
            if name == 'metric':
                observed_metrics.append(float(msg.data))
            if name == 'gt' and args.mode == 'live' and len(captured) < 10:
                key = stamp(msg)
                if not any(item['stamp'] == key for item in captured):
                    name = f'live_gt_{len(captured):02d}.bin'
                    (output / name).write_bytes(bytes(msg.data))
                    captured.append({'file': name, 'format': msg.format, 'stamp': key})
        for name, topic, kind in [('rgb', rgb_topic, CompressedImage), ('gt', gt_topic, CompressedImage),
                                  ('prediction', prediction_topic, Image), ('metric', metric_topic, Float32)]:
            subscriptions.append(node.create_subscription(kind, topic, lambda msg, n=name: receive(n, msg), qos if name != 'metric' else 10))
        publishers = [node.create_publisher(CompressedImage, topic, qos) for topic in (rgb_topic, gt_topic)] if args.mode == 'replay' else []
        parameters = {
            'inference': {'checkpoint_path': str(checkpoint), 'device': args.device, 'input_topic': rgb_topic,
                          'mask_topic': prediction_topic, 'color_topic': prefix + '/color'},
            'evaluation': {'prediction_topic': prediction_topic, 'ground_truth_topic': gt_topic,
                           'frame_miou_topic': prefix + '/frame_miou', 'cumulative_miou_topic': metric_topic,
                           'per_class_iou_topic': prefix + '/per_class', 'ground_truth_color_topic': prefix + '/gt_color',
                           'overlay_topic': prefix + '/overlay'}}
        for role, values in parameters.items():
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', role, '--output', str(output), '--ros-args']
            for key, value in values.items():
                cmd.extend(['-p', f'{key}:={value}'])
            log = (output / f'{role}.log').open('w', encoding='utf-8')
            logs.append(log)
            processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))

        def spin_until(condition, timeout):
            end = time.monotonic() + timeout
            while not condition():
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('A ROS worker exited early; inspect saved worker logs')
                if time.monotonic() >= end:
                    raise TimeoutError('ROS experiment timed out; inspect middleware, source and worker logs')
                rclpy.spin_once(node, timeout_sec=0.05)

        if args.mode == 'replay':
            spin_until(lambda: all(p.get_subscription_count() >= 2 for p in publishers)
                       and node.count_publishers(metric_topic) == 1 and node.count_publishers(prediction_topic) == 1, 60)
            started = time.monotonic()
            for index, (rgb_path, label_path) in enumerate(pairs):
                rgb, gt, _ = encode_pair(rgb_path, label_path)
                key = (100 + index, 0)
                for publisher, data, fmt in zip(publishers, (rgb, gt), ('jpeg', 'png')):
                    msg = CompressedImage()
                    msg.header.stamp.sec, msg.header.stamp.nanosec = key
                    msg.format, msg.data = fmt, data
                    publisher.publish(msg)
                spin_until(lambda: key in observed_masks and len(observed_metrics) >= index + 1, 30)
                np.save(output / f'prediction_{index:06d}.npy', observed_masks.pop(key))
            spin_until(lambda: (output / 'pairs.jsonl').is_file()
                       and len((output / 'pairs.jsonl').read_text().splitlines()) == 10, 10)
        else:
            spin_until(lambda: len(observed_metrics) > 0, 60)
            started = time.monotonic()
            end = started + args.seconds
            spin_until(lambda: time.monotonic() >= end, args.seconds + 5)
        success = True
    finally:
        elapsed = time.monotonic() - started if started is not None else 0
        for process in processes:
            stop_process(process)
        for log in logs:
            log.close()
        success = success and all(p.returncode == 0 for p in processes)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        rates = {key: {'received': len(times), 'window_seconds': elapsed,
                       'hz': len(times) / elapsed if elapsed else None} for key, times in measurements.items()}
        save_json(output / 'observation.json', {'completed': success, 'rates': rates, 'captured_gt': captured,
                                               'received_cumulative_miou': observed_metrics,
                                               'worker_exit_codes': [p.returncode for p in processes]})
    final = json.loads((output / 'evaluation_final.json').read_text())
    if any(p.returncode != 0 for p in processes):
        raise RuntimeError('Worker exited with an error; inspect logs before accepting results')
    if not final['pair_count'] or (args.mode == 'replay' and final['pair_count'] != 10):
        raise AssertionError(f'Unexpected processed pair count: {final["pair_count"]}')
    print(json.dumps({'output': str(output), 'pairs': final['pair_count'], 'rates': rates}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['inference', 'evaluation'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['replay', 'live'], default='replay')
    parser.add_argument('--checkpoint')
    parser.add_argument('--samples')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--rgb-topic', default='/carla/ego_vehicle/rgb/image/compressed')
    parser.add_argument('--gt-topic', default='/carla/ego_vehicle/segmentation/image/compressed')
    arguments, ros_arguments = parser.parse_known_args()
    if arguments.worker:
        worker(arguments.worker, Path(arguments.output), ros_arguments)
    else:
        if not 1 <= arguments.seconds <= 600:
            parser.error('--seconds must be in [1, 600]')
        if ros_arguments:
            parser.error(f'Unknown arguments: {ros_arguments}')
        run(arguments)
