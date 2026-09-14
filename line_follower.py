"""蓝线识别、PID 循迹和丢线搜索状态机。"""

import time

import cv2
import numpy as np

from config import BLUE_LOWER, BLUE_UPPER
from pid_controller import EnhancedPID


class BlueLineFollower:
    def __init__(self):
        self.pid = EnhancedPID(p=0.4, i=0.001, d=0.05, out_limit=80)
        self.lower_blue = np.array(BLUE_LOWER)
        self.upper_blue = np.array(BLUE_UPPER)
        self.frame_center = None
        self.line_lost_count = 0
        self.max_line_lost = 10
        self.search_state = "idle"
        self.search_start_time = 0
        self.position_history = []
        self.history_length = 8

    def reset_after_interruption(self):
        self.pid.clear()
        self.position_history.clear()

    def process(self, frame, ep_chassis):
        height, width = frame.shape[:2]
        if self.frame_center is None:
            self.frame_center = width // 2
            print(f"检测到图像尺寸: 宽度={width}, 高度={height}")
            print(f"图像中心点: {self.frame_center}")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        roi_rows = [
            height - 400,
            height - 300,
            height - 250,
            height - 200,
            height - 100,
            height - 70,
            height - 40,
            height - 20,
        ]
        line_centers = []
        for row in roi_rows:
            if row < 0 or row >= height:
                continue
            white_pixels = np.where(mask[row, :] == 255)[0]
            if len(white_pixels) > 10:
                center = int(np.mean(white_pixels))
                line_centers.append(center)
                cv2.line(frame, (0, row), (width, row), (0, 255, 0), 1)
                cv2.circle(frame, (center, row), 5, (0, 0, 255), -1)

        if line_centers:
            self._follow_line(frame, ep_chassis, line_centers)
        else:
            self._search_for_line(frame, ep_chassis)
        return mask

    def _follow_line(self, frame, ep_chassis, line_centers):
        self.search_state = "idle"
        self.line_lost_count = 0
        self.position_history.append(int(np.mean(line_centers)))
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)

        center = int(np.mean(self.position_history))
        error = self.frame_center - center
        z_speed = self.pid.update(0, error)
        x_speed = 0.5 - min(abs(error) / 100.0, 0.2)
        ep_chassis.drive_speed(x=x_speed, y=0, z=z_speed, timeout=0.1)

        height = frame.shape[0]
        cv2.circle(frame, (center, height - 50), 8, (255, 0, 0), -1)
        cv2.circle(frame, (self.frame_center, height - 50), 8, (0, 255, 255), -1)
        cv2.putText(frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Z-Speed: {z_speed:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _search_for_line(self, frame, ep_chassis):
        self.line_lost_count += 1
        if self.line_lost_count <= self.max_line_lost:
            return

        now = time.time()
        if self.search_state == "idle":
            self.search_state = "left_turn_first"
            self.search_start_time = now
            print("开始寻找蓝线")
            return

        commands = {
            "left_turn_first": (30, 2.0, "check_after_left", "第一次左转完成"),
            "check_after_left": (0, 1.0, "right_turn_first", "左转后检查完成"),
            "right_turn_first": (-30, 2.0, "right_turn_second", "第一次右转完成"),
            "right_turn_second": (-30, 2.0, "check_after_rights", "第二次右转完成"),
            "check_after_rights": (0, 1.0, "left_turn_final", "右转后检查完成"),
            "left_turn_final": (30, 2.0, "move_forward", "最后左转完成"),
        }

        if self.search_state in commands:
            z_speed, duration, next_state, message = commands[self.search_state]
            ep_chassis.drive_speed(x=0, y=0, z=z_speed, timeout=0.1)
            cv2.putText(
                frame,
                f"SEARCHING: {self.search_state}",
                (frame.shape[1] // 2 - 180, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            if now - self.search_start_time > duration:
                self.search_state = next_state
                self.search_start_time = now
                print(message)
            return

        if self.search_state == "move_forward":
            ep_chassis.drive_speed(x=0.2, y=0, z=0, timeout=0.1)
            cv2.putText(
                frame,
                "SEARCHING: Moving forward 0.1m",
                (frame.shape[1] // 2 - 170, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            if now - self.search_start_time > 1.0:
                self.search_state = "left_turn_first"
                self.search_start_time = now
                print("前进完成")
