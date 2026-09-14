"""数字标记检测、云台对准和拍照。"""

import os
import threading
import time

import cv2

from config import (
    MARKER_CENTER_THRESHOLD,
    MARKER_MIN_HEIGHT,
    MARKER_MIN_WIDTH,
    MARKER_TRACK_TIMEOUT,
    TARGET_MARKERS,
    ensure_photo_dir,
)


class MarkerInfo:
    """RoboMaster 视觉模块返回的归一化标记信息。"""

    def __init__(self, x, y, w, h, info):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.info = info

    @property
    def pt1(self):
        return int((self.x - self.w / 2) * 1280), int((self.y - self.h / 2) * 720)

    @property
    def pt2(self):
        return int((self.x + self.w / 2) * 1280), int((self.y + self.h / 2) * 720)

    @property
    def center(self):
        return int(self.x * 1280), int(self.y * 720)

    @property
    def text(self):
        return self.info


class MarkerDetector:
    def __init__(self):
        self.markers = []
        self.lock = threading.Lock()
        self.detected_targets = set()
        self.photo_taken_targets = set()

    def on_detect_marker(self, marker_info):
        """作为 RoboMaster marker 订阅的回调函数。"""
        with self.lock:
            self.markers = [MarkerInfo(*item) for item in marker_info]

    def track_and_take_photo(
        self,
        ep_gimbal,
        ep_camera,
        target_text,
        center_threshold=MARKER_CENTER_THRESHOLD,
    ):
        kp_yaw = 20.0
        kp_pitch = 30.5
        kp_gimbal_yaw = 20.0
        deadzone_x = 0.003
        deadzone_y = 0.003

        start_time = time.time()
        save_dir = ensure_photo_dir()

        while time.time() - start_time < MARKER_TRACK_TIMEOUT:
            image = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if image is None:
                continue

            with self.lock:
                target = next((m for m in self.markers if m.text == target_text), None)

            if target is None:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                cv2.putText(
                    image,
                    f"Target marker {target_text} not found",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.rectangle(image, target.pt1, target.pt2, (255, 255, 255))
                cv2.putText(
                    image,
                    target.text,
                    target.center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    3,
                )
                cv2.circle(image, target.center, 10, (0, 0, 255), -1)
                cv2.circle(image, target.center, 12, (255, 0, 0), 2)

                error_x = target.x - 0.5
                error_y = target.y - 0.5
                error_x = 0 if abs(error_x) < deadzone_x else error_x
                error_y = 0 if abs(error_y) < deadzone_y else error_y

                z_speed = -kp_yaw * error_x
                pitch_speed = max(-60, min(60, -kp_pitch * error_y))
                yaw_speed = max(-60, min(60, -kp_gimbal_yaw * z_speed))
                ep_gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=yaw_speed)

                marker_x, marker_y = target.center
                distance = ((marker_x - 640) ** 2 + (marker_y - 360) ** 2) ** 0.5
                if distance < center_threshold:
                    self._save_marker_photo(image, target.text, save_dir)
                    ep_gimbal.moveto(
                        pitch=-30, yaw=0, pitch_speed=100, yaw_speed=100
                    ).wait_for_completed()
                    time.sleep(1)
                    return True

            cv2.line(image, (640, 0), (640, 720), (0, 255, 0), 2)
            cv2.line(image, (0, 360), (1280, 360), (0, 255, 0), 2)
            cv2.imshow("Marker Tracking", image)
            if cv2.waitKey(1) == ord("q"):
                break

        return False

    @staticmethod
    def _save_marker_photo(image, marker_text, save_dir):
        cv2.line(image, (640, 0), (640, 720), (0, 255, 0), 2)
        cv2.line(image, (0, 360), (1280, 360), (0, 255, 0), 2)
        text = f"Team 8 detects marker with ID of {marker_text}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        cv2.putText(image, text, (text_x, 50), font, 1.5, (0, 255, 255), 3)

        filename = f"marker_photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(save_dir, filename)
        cv2.imwrite(path, image)
        print(f"Photo has taken! Save photo to: {path}")

    def process(self, ep_chassis, ep_gimbal, ep_camera, image):
        """发现新的 1～5 号标记时停车、对准并拍照。"""
        current_target = None
        with self.lock:
            marker_snapshot = list(self.markers)

        for marker in marker_snapshot:
            cv2.rectangle(image, marker.pt1, marker.pt2, (255, 255, 255))
            cv2.putText(
                image,
                marker.text,
                marker.center,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )
            if (
                marker.text in TARGET_MARKERS
                and marker.text not in self.detected_targets
                and marker.text not in self.photo_taken_targets
                and marker.w > MARKER_MIN_WIDTH
                and marker.h > MARKER_MIN_HEIGHT
            ):
                current_target = marker.text
                self.detected_targets.add(marker.text)

        if current_target is None:
            return False

        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        time.sleep(1)
        print(f"Group 8, detect target marker: {current_target}, stop!")

        photo_taken = self.track_and_take_photo(ep_gimbal, ep_camera, current_target)
        if photo_taken:
            self.photo_taken_targets.add(current_target)
            ep_gimbal.moveto(
                pitch=-30, yaw=0, pitch_speed=100, yaw_speed=100
            ).wait_for_completed()
            time.sleep(1)
        else:
            print("Photo timeout or failure, continue patrolling the line")
        return True
