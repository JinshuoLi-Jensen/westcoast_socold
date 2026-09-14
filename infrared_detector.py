"""红外测距避障：发现另一台机器人后停车并拍照。"""

import os
import time
from datetime import datetime

import cv2

from config import (
    IR_LOWER_THRESHOLD,
    IR_PHOTO_COOLDOWN,
    IR_UPPER_THRESHOLD,
    ensure_photo_dir,
)


class InfraredDetector:
    def __init__(self, ep_robot):
        self.ep_sensor = ep_robot.sensor
        self.ir_upper_threshold = IR_UPPER_THRESHOLD
        self.ir_lower_threshold = IR_LOWER_THRESHOLD
        self.detected_condition = False
        self.last_photo_time = 0
        self.photo_cooldown = IR_PHOTO_COOLDOWN
        self.has_taken_photo_for_current_obstacle = False
        self.fully_stopped = False
        self.stop_start_time = 0
        self.ir_distance = 0

        self.ep_sensor.sub_distance(freq=5, callback=self.ir_sensor_callback)
        self.photo_dir = ensure_photo_dir()

    def ir_sensor_callback(self, sub_info):
        self.ir_distance = sub_info[0]

    def take_photo(self, frame):
        current_time = time.time()
        if current_time - self.last_photo_time < self.photo_cooldown:
            return False
        self.last_photo_time = current_time

        cv2.putText(
            frame,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            f"IR: {self.ir_distance}cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        text = "Team 8 detected a robot"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_width = cv2.getTextSize(text, font, 1.5, 3)[0][0]
        cv2.putText(
            frame,
            text,
            ((frame.shape[1] - text_width) // 2, 50),
            font,
            1.5,
            (0, 255, 255),
            3,
        )
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (90, 70), (width - 90, height - 70), (0, 0, 255), 10)

        filename = os.path.join(
            self.photo_dir,
            f"robot_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
        )
        cv2.imwrite(filename, frame)
        print(f"已保存机器人检测照片: {filename}")
        return True

    def process_detection(self, ep_chassis, frame):
        current_time = time.time()
        if self.ir_distance <= self.ir_lower_threshold:
            ep_chassis.drive_speed(x=0, y=0, z=0)
            if not self.detected_condition:
                self.detected_condition = True
                self.fully_stopped = False
                self.stop_start_time = current_time
                print("检测到另一个机器人，开始停止")

            if not self.fully_stopped and current_time - self.stop_start_time > 0.5:
                self.fully_stopped = True
                print("机器人已完全停止")

            if self.fully_stopped and not self.has_taken_photo_for_current_obstacle:
                if self.take_photo(frame):
                    print("已拍摄另一个机器人的照片")
                self.has_taken_photo_for_current_obstacle = True
            return True

        if self.ir_distance >= self.ir_upper_threshold:
            self.has_taken_photo_for_current_obstacle = False
            self.detected_condition = False
            self.fully_stopped = False
            return False

        if self.detected_condition:
            ep_chassis.drive_speed(x=0, y=0, z=0)
        return self.detected_condition
