import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='mini_dpvo.*')
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation as R
from mini_dpvo.dpvo import DPVO  # ty:ignore[unresolved-import]
from mini_dpvo.config import cfg as base_cfg  # ty:ignore[unresolved-import]

_config_path = Path(get_package_share_directory('dpvo')) / 'config' / 'config.yaml'
_config = yaml.safe_load(_config_path.read_text())
NETWORK_PATH = _config['network_path']
CONFIG_PATH = _config['dpvo_config_path']
FX, FY = _config['fx'], _config['fy']
CX, CY = _config['cx'], _config['cy']
DOWNSCALE = _config['downscale']
VIDEO_TOPIC = _config['video_topic']

# a post-BA edge counts as an inlier if its reprojection residual (network-res pixels)
# is below this; tune against the raw error on /dpvo/ba_reproj_error
INLIER_PX = 1.0

# change of basis from the OpenCV optical convention (x right, y down, z forward)
# to the ROS REP-103 convention (x forward, y left, z up): v_ros = CV_TO_ROS @ v_cv
CV_TO_ROS = np.eye(4)
CV_TO_ROS[:3, :3] = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])

class DpvoNode(Node):
    def __init__(self):
        super().__init__('dpvo_test')
        self.bridge = CvBridge()

        base_cfg.merge_from_file(CONFIG_PATH)
        self.cfg = base_cfg

        # ponytail: BUFFER_SIZE=2048 keyframe ceiling, fine for a test run;
        # for long missions DPVO needs a sliding buffer it doesn't have
        self.slam = None  # created on first frame (needs image size)
        self.frame_count = 0

        self.prev_status = ""

        self.path = PoseArray()
        self.path.header.frame_id = 'odom'

        self.video_subscription = self.create_subscription(Image, VIDEO_TOPIC, self.camera_cb, 1)
        self.path_publisher = self.create_publisher(PoseArray, '/monovo/path', 10)
        self.confidence_publisher = self.create_publisher(Float32, '/monovo/ba_confidence', 10)

    @torch.no_grad()
    def camera_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        bgr = cv2.resize(bgr, None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA) # pyrefly: ignore[bad-argument-type]
        h, w, _ = bgr.shape
        bgr = bgr[: h - h % 16, : w - w % 16]

        image = torch.from_numpy(bgr).permute(2, 0, 1).cuda()
        intrinsics = torch.tensor([FX, FY, CX, CY], dtype=torch.float64).cuda() * DOWNSCALE

        if self.slam is None:
            self.slam = DPVO(self.cfg, NETWORK_PATH, ht=image.shape[1], wd=image.shape[2])

        self.slam(self.frame_count, image, intrinsics)
        self.frame_count += 1

        # DPVO extracts a FIXED patch budget at random locations, so a raw feature
        # count is constant and can't flag sparse scenes; the median residual is fooled
        # too (BA fits the good few, the median ignores the bad tail). Instead publish
        # the inlier fraction -- share of the budget whose post-BA residual clears
        # INLIER_PX -- which drops when most of the scene is uninformative. [0, 1],
        # higher == better, density-aware. 0.0 before BA runs.
        err = self.slam.ba_rproj_error
        if err is not None and err.numel():
            confidence = float((err < INLIER_PX).float().mean().item())
        else:
            confidence = 0.0
        self.confidence_publisher.publish(Float32(data=confidence))

        if not self.slam.is_initialized:
            if self.prev_status != "initializing":
                self.prev_status = "initializing"
                self.get_logger().info(f'status: {self.prev_status}')
            return

        # last non-zero pose is cam_T_world (see inference.log_trajectory); invert to world_T_cam
        poses = self.slam.poses_
        nonzero = poses[~(poses[:, :6] == 0).all(dim=1)]
        quat_pose = nonzero[-1].numpy(force=True)
        cam_T_world = np.eye(4)
        cam_T_world[:3, :3] = R.from_quat(quat_pose[3:]).as_matrix()
        cam_T_world[:3, 3] = quat_pose[:3]
        world_T_cam = np.linalg.inv(cam_T_world)

        # re-express both the world frame and the camera body frame in ROS axes
        world_T_cam = CV_TO_ROS @ world_T_cam @ CV_TO_ROS.T

        ros_pose = Pose()
        ros_pose.position.x, ros_pose.position.y, ros_pose.position.z = (float(v) for v in world_T_cam[:3, 3])
        quat_xyzw = R.from_matrix(world_T_cam[:3, :3]).as_quat()
        ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w = (
            float(v) for v in quat_xyzw
        )
        self.path.poses.append(ros_pose)

        self.path.header.stamp = msg.header.stamp
        self.path_publisher.publish(self.path)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = DpvoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()