import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from .carla_labels import (
    NUM_CLASSES,
    label_to_color_image,
)
from .model import LightUNet
from .utils import (
    decode_compressed_image,
    preprocess_rgb,
)


class SegmentationInferenceNode(Node):
    def __init__(self):
        super().__init__("segmentation_inference_node")

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter(
            "input_topic",
            "/carla/ego_vehicle/rgb/image/compressed",
        )
        self.declare_parameter(
            "mask_topic",
            "/acdcrp/prediction/mask",
        )
        self.declare_parameter(
            "color_topic",
            "/acdcrp/prediction/color",
        )
        self.declare_parameter("height", 256)
        self.declare_parameter("width", 512)
        self.declare_parameter("publish_color", True)
        self.declare_parameter("log_every_n", 30)

        checkpoint_path = self.get_parameter(
            "checkpoint_path"
        ).value
        requested_device = self.get_parameter("device").value
        input_topic = self.get_parameter("input_topic").value
        mask_topic = self.get_parameter("mask_topic").value
        color_topic = self.get_parameter("color_topic").value

        self.height = int(
            self.get_parameter("height").value
        )
        self.width = int(
            self.get_parameter("width").value
        )
        self.publish_color = bool(
            self.get_parameter("publish_color").value
        )
        self.log_every_n = int(
            self.get_parameter("log_every_n").value
        )

        if not checkpoint_path:
            raise ValueError(
                "checkpoint_path parameter is required"
            )

        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        if (
            str(requested_device).startswith("cuda")
            and not torch.cuda.is_available()
        ):
            self.get_logger().warning(
                "CUDA requested but unavailable; using CPU"
            )
            requested_device = "cpu"

        self.device = torch.device(requested_device)

        self.model = LightUNet(
            num_classes=NUM_CLASSES
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.bridge = CvBridge()
        self.frame_count = 0

        self.mask_publisher = self.create_publisher(
            Image,
            mask_topic,
            qos_profile_sensor_data,
        )

        self.color_publisher = self.create_publisher(
            Image,
            color_topic,
            qos_profile_sensor_data,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            input_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Loaded checkpoint: {checkpoint_path}"
        )
        self.get_logger().info(
            f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}"
        )
        self.get_logger().info(
            f"Device: {self.device}"
        )
        self.get_logger().info(
            f"Input topic: {input_topic}"
        )
        self.get_logger().info(
            f"Prediction size: {self.width}x{self.height}"
        )

    def image_callback(self, message):
        start_time = time.perf_counter()

        try:
            image_rgb = decode_compressed_image(message)

            input_tensor = preprocess_rgb(
                image_rgb,
                height=self.height,
                width=self.width,
            ).to(self.device)

            with torch.inference_mode():
                logits = self.model(input_tensor)
                prediction = (
                    logits.argmax(dim=1)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

            mask_message = self.bridge.cv2_to_imgmsg(
                prediction,
                encoding="mono8",
            )
            mask_message.header = message.header
            self.mask_publisher.publish(mask_message)

            if self.publish_color:
                color_image = label_to_color_image(
                    prediction
                )

                color_message = self.bridge.cv2_to_imgmsg(
                    color_image,
                    encoding="rgb8",
                )
                color_message.header = message.header
                self.color_publisher.publish(color_message)

            self.frame_count += 1

            if (
                self.log_every_n > 0
                and self.frame_count % self.log_every_n == 0
            ):
                elapsed = time.perf_counter() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                self.get_logger().info(
                    f"Frame {self.frame_count}: "
                    f"inference={elapsed * 1000:.1f} ms, "
                    f"processing_fps={fps:.2f}, "
                    f"classes={np.unique(prediction).tolist()}"
                )

        except Exception as error:
            self.get_logger().error(
                f"Inference callback failed: {error}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = SegmentationInferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()