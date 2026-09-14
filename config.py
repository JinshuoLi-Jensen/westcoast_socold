"""集中保存原程序中的可调参数。"""

import os

_ORIGINAL_PHOTO_DIR = r"D:\robotest"
_LOCAL_PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos")

# 可用 ROBOT_PHOTO_DIR 自定义；没有 D 盘时自动使用当前程序旁的 photos 文件夹。
PHOTO_DIR = os.environ.get(
    "ROBOT_PHOTO_DIR",
    _ORIGINAL_PHOTO_DIR if os.path.isdir("D:\\") else _LOCAL_PHOTO_DIR,
)

# 数字标记
TARGET_MARKERS = {"1", "2", "3", "4", "5"}
MARKER_MIN_WIDTH = 0.1
MARKER_MIN_HEIGHT = 0.1
MARKER_CENTER_THRESHOLD = 5
MARKER_TRACK_TIMEOUT = 10

# 蓝线 HSV 范围
BLUE_LOWER = (100, 80, 50)
BLUE_UPPER = (130, 255, 255)

# 红外测距阈值
IR_LOWER_THRESHOLD = 340
IR_UPPER_THRESHOLD = 500
IR_PHOTO_COOLDOWN = 10.0


def ensure_photo_dir():
    """确保拍照目录存在并返回该目录。"""
    os.makedirs(PHOTO_DIR, exist_ok=True)
    return PHOTO_DIR
