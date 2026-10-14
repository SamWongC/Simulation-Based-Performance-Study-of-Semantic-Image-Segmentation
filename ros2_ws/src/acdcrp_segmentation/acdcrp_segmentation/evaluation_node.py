import numpy as np
import message_filters
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Float32MultiArray

from .carla_labels import (
    NUM_CLASSES,
    REDUCED_CLASSES,
    label_to_color_image,
)
from .utils import (
    compute_intersection_union,
    compute_iou_from_counts,
    decode_ground_truth,
)


class SegmentationEvaluationNode(Node):
    def __init__(self):
        super().__init__(
            "segmentation_evaluation_node"
        )

        self.declare_parameter(
            "prediction_topic",
            "/acdcrp/prediction/mask",
        )
        self.declare_parameter(
            "ground_truth_topic",
            "/carla/ego_vehicle/"
            "segmentation/image/compressed",
        )
        self.declare_parameter(
            "frame_miou_topic",
            "/acdcrp/evaluation/frame_miou",
        )
        self.declare_parameter(
            "cumulative_miou_topic",
            "/acdcrp/evaluation/cumulative_miou",
        )
        self.declare_parameter(
            "per_class_iou_topic",
            "/acdcrp/evaluation/per_class_iou",
        )
        self.declare_parameter(
            "ground_truth_color_topic",
            "/acdcrp/evaluation/ground_truth_color",
        )
        self.declare_parameter(
            "overlay_topic",
            "/acdcrp/evaluation/overlay",
        )
        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("sync_slop", 0.15)
        self.declare_parameter("log_every_n", 30)

        prediction_topic = self.get_parameter(
            "prediction_topic"
        ).value
        ground_truth_topic = self.get_parameter(
            "ground_truth_topic"
        ).value

        frame_miou_topic = self.get_parameter(
            "frame_miou_topic"
        ).value
        cumulative_miou_topic = self.get_parameter(
            "cumulative_miou_topic"
        ).value
        per_class_iou_topic = self.get_parameter(
            "per_class_iou_topic"
        ).value
        ground_truth_color_topic = self.get_parameter(
            "ground_truth_color_topic"
        ).value
        overlay_topic = self.get_parameter(
            "overlay_topic"
        ).value

        sync_queue_size = int(
            self.get_parameter(
                "sync_queue_size"
            ).value
        )
        sync_slop = float(
            self.get_parameter("sync_slop").value
        )
        self.log_every_n = int(
            self.get_parameter("log_every_n").value
        )

        self.bridge = CvBridge()

        self.total_intersection = np.zeros(
            NUM_CLASSES,
            dtype=np.int64,
        )
        self.total_union = np.zeros(
            NUM_CLASSES,
            dtype=np.int64,
        )
        self.pair_count = 0

        self.frame_miou_publisher = (
            self.create_publisher(
                Float32,
                frame_miou_topic,
                10,
            )
        )

        self.cumulative_miou_publisher = (
            self.create_publisher(
                Float32,
                cumulative_miou_topic,
                10,
            )
        )

        self.per_class_iou_publisher = (
            self.create_publisher(
                Float32MultiArray,
                per_class_iou_topic,
                10,
            )
        )

        self.ground_truth_color_publisher = (
            self.create_publisher(
                Image,
                ground_truth_color_topic,
                qos_profile_sensor_data,
            )
        )

        self.overlay_publisher = (
            self.create_publisher(
                Image,
                overlay_topic,
                qos_profile_sensor_data,
            )
        )

        self.prediction_subscriber = (
            message_filters.Subscriber(
                self,
                Image,
                prediction_topic,
                qos_profile=qos_profile_sensor_data,
            )
        )

        self.ground_truth_subscriber = (
            message_filters.Subscriber(
                self,
                CompressedImage,
                ground_truth_topic,
                qos_profile=qos_profile_sensor_data,
            )
        )

        self.synchronizer = (
            message_filters.ApproximateTimeSynchronizer(
                [
                    self.prediction_subscriber,
                    self.ground_truth_subscriber,
                ],
                queue_size=sync_queue_size,
                slop=sync_slop,
            )
        )

        self.synchronizer.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            f"Prediction topic: {prediction_topic}"
        )
        self.get_logger().info(
            f"Ground-truth topic: "
            f"{ground_truth_topic}"
        )
        self.get_logger().info(
            f"Synchronization: queue={sync_queue_size}, "
            f"slop={sync_slop:.3f} s"
        )
        self.get_logger().info(
            "Class order: "
            + ", ".join(
                f"{index}={name}"
                for index, name
                in REDUCED_CLASSES.items()
            )
        )

    def synchronized_callback(
        self,
        prediction_message,
        ground_truth_message,
    ):
        try:
            prediction = (
                self.bridge.imgmsg_to_cv2(
                    prediction_message,
                    desired_encoding="mono8",
                )
            )

            prediction = np.asarray(
                prediction,
                dtype=np.uint8,
            )

            if prediction.ndim == 3:
                prediction = prediction.squeeze()

            ground_truth = decode_ground_truth(
                ground_truth_message,
                height=prediction.shape[0],
                width=prediction.shape[1],
            )

            frame_intersection, frame_union = (
                compute_intersection_union(
                    prediction,
                    ground_truth,
                    num_classes=NUM_CLASSES,
                    ignore_index=255,
                )
            )

            frame_per_class_iou, frame_miou = (
                compute_iou_from_counts(
                    frame_intersection,
                    frame_union,
                )
            )

            self.total_intersection += (
                frame_intersection
            )
            self.total_union += frame_union

            (
                cumulative_per_class_iou,
                cumulative_miou,
            ) = compute_iou_from_counts(
                self.total_intersection,
                self.total_union,
            )

            self.frame_miou_publisher.publish(
                Float32(data=float(frame_miou))
            )

            self.cumulative_miou_publisher.publish(
                Float32(
                    data=float(cumulative_miou)
                )
            )

            per_class_values = np.nan_to_num(
                cumulative_per_class_iou,
                nan=-1.0,
            ).astype(np.float32)

            self.per_class_iou_publisher.publish(
                Float32MultiArray(
                    data=per_class_values.tolist()
                )
            )

            ground_truth_color = (
                label_to_color_image(
                    ground_truth
                )
            )

            ground_truth_color_message = (
                self.bridge.cv2_to_imgmsg(
                    ground_truth_color,
                    encoding="rgb8",
                )
            )
            ground_truth_color_message.header = (
                ground_truth_message.header
            )

            self.ground_truth_color_publisher.publish(
                ground_truth_color_message
            )

            overlay = self.create_error_overlay(
                prediction,
                ground_truth,
            )

            overlay_message = (
                self.bridge.cv2_to_imgmsg(
                    overlay,
                    encoding="rgb8",
                )
            )
            overlay_message.header = (
                prediction_message.header
            )

            self.overlay_publisher.publish(
                overlay_message
            )

            self.pair_count += 1

            if (
                self.log_every_n > 0
                and self.pair_count
                % self.log_every_n == 0
            ):
                per_class_text = ", ".join(
                    f"{REDUCED_CLASSES[index]}="
                    f"{value:.3f}"
                    if not np.isnan(value)
                    else
                    f"{REDUCED_CLASSES[index]}=N/A"
                    for index, value in enumerate(
                        cumulative_per_class_iou
                    )
                )

                self.get_logger().info(
                    f"Pair {self.pair_count}: "
                    f"frame_mIoU={frame_miou:.4f}, "
                    f"cumulative_mIoU="
                    f"{cumulative_miou:.4f}"
                )
                self.get_logger().info(
                    f"Per-class IoU: "
                    f"{per_class_text}"
                )

        except Exception as error:
            self.get_logger().error(
                f"Evaluation callback failed: "
                f"{error}"
            )

    @staticmethod
    def create_error_overlay(
        prediction,
        ground_truth,
    ):
        prediction_color = label_to_color_image(
            prediction
        )

        overlay = prediction_color.copy()

        valid = ground_truth != 255
        incorrect = (
            prediction != ground_truth
        ) & valid

        overlay[incorrect] = np.array(
            [255, 0, 0],
            dtype=np.uint8,
        )

        overlay[~valid] = np.array(
            [0, 0, 0],
            dtype=np.uint8,
        )

        return overlay


def main(args=None):
    rclpy.init(args=args)

    node = SegmentationEvaluationNode()

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