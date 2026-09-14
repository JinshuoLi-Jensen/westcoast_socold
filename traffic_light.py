"""基于 HSV 颜色、面积、半径和圆形度的红绿灯检测。"""

import math
import os
import time

import cv2
import numpy as np

from config import ensure_photo_dir


class TrafficLightDetector:
    def __init__(self):
        self.is_waiting = False
        self.last_detection_time = 0
        self.cooldown = 0.5
        self.image_count = 0
        self.last_light_color = None
        self.img_width = 640
        self.img_height = 480

        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([8, 255, 255])
        self.red_lower2 = np.array([172, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        self.green_lower = np.array([50, 80, 80])
        self.green_upper = np.array([70, 255, 255])

        self.min_area = 100
        self.circularity_threshold = 0.79
        self.min_radius = 12
        print("The traffic light detector is initialized.")

    @staticmethod
    def calculate_circularity(contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area == 0 or perimeter == 0:
            return 0
        return 4 * math.pi * area / (perimeter * perimeter)

    @staticmethod
    def calculate_radius(contour):
        area = cv2.contourArea(contour)
        return math.sqrt(area / math.pi) if area else 0

    def _best_valid_contour(self, mask, current_best):
        best_circularity, best_contour = current_best
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) <= self.min_area:
                continue
            if self.calculate_radius(contour) < self.min_radius:
                continue
            circularity = self.calculate_circularity(contour)
            if circularity > self.circularity_threshold and circularity > best_circularity:
                best_circularity, best_contour = circularity, contour
        return best_circularity, best_contour

    def detect_light(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.red_lower1, self.red_upper1),
            cv2.inRange(hsv, self.red_lower2, self.red_upper2),
        )
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        red_score, red_contour = self._best_valid_contour(red_mask, (0, None))
        green_score, green_contour = self._best_valid_contour(green_mask, (0, None))
        if red_score == 0 and green_score == 0:
            return None, None
        if red_score >= green_score:
            return "red", red_contour
        return "green", green_contour

    def save_detection_image(self, frame, color, contour):
        if contour is None or color == self.last_light_color:
            return

        annotated = frame.copy()
        cv2.drawContours(annotated, [contour], -1, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(contour)
        label = f"Team 8 detects a {color} light"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(label, font, 0.8, 2)
        text_x = x + (w - text_width) // 2
        text_y = y + h + text_height + 5

        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        cv2.putText(annotated, label, (text_x, text_y), font, 0.8, (0, 255, 255), 2)

        filename = (
            f"traffic_light_{color}_{time.strftime('%Y%m%d_%H%M%S')}_"
            f"{self.image_count:04d}.jpg"
        )
        cv2.imwrite(os.path.join(ensure_photo_dir(), filename), annotated)
        self.image_count += 1
        self.last_light_color = color
        print(f"saved image: {filename}")

    def process_frame(self, frame, ep_chassis):
        if frame is None:
            return

        current_time = time.time()
        if current_time - self.last_detection_time < self.cooldown:
            return

        resized = cv2.resize(frame, (self.img_width, self.img_height))
        color, contour = self.detect_light(resized)
        self.last_detection_time = current_time
        if color:
            self.save_detection_image(resized, color, contour)

        if color == "red":
            ep_chassis.drive_speed(x=0, y=0, z=0)
            self.is_waiting = True
            print("red light detected, stop moving")
        elif color == "green" and self.is_waiting:
            self.is_waiting = False
            print("green light detected, moving")
