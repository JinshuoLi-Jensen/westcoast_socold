"""Task 5: infrared-triggered obstacle avoidance for RoboMaster EP.

The class is deliberately independent of the line follower.  While an obstacle
is being handled it owns the chassis; after the manoeuvre it returns False so
the normal line-following loop can resume.
"""

import os
import time
from datetime import datetime

import cv2


class ObstacleAvoider:
    """Stop, choose a side, pass the obstacle, and return to the line."""

    def __init__(self, ep_robot, lower_threshold=340, upper_threshold=500,
                 lateral_speed=0.35, forward_speed=0.35):
        self.sensor = ep_robot.sensor
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.lateral_speed = lateral_speed
        self.forward_speed = forward_speed
        self.distance = upper_threshold + 1
        self.state = "normal"
        self.state_started = 0.0
        self.side = None
        self.photo_saved = False
        self.armed = True
        self.photo_cooldown = 10.0
        self.last_photo_time = 0.0
        self.photo_dir = os.environ.get(
            "ROBOT_PHOTO_DIR", os.path.join(os.path.dirname(__file__), "photos"))
        os.makedirs(self.photo_dir, exist_ok=True)
        self.sensor.sub_distance(freq=10, callback=self._distance_callback)

    def _distance_callback(self, values):
        if values:
            self.distance = float(values[0])

    def _choose_side(self, frame):
        """Estimate the clearer side from the lower camera region.

        The obstacle is normally centred on the coloured route.  A lower edge
        density is treated as the clearer passage.  If the image is unavailable
        or tied, the left side is selected deterministically.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return "left"
        h, w = frame.shape[:2]
        roi = cv2.cvtColor(frame[int(h * 0.45):, :], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(roi, 60, 150)
        mid = w // 2
        left_score = float(edges[:, :mid].mean()) if mid else 0.0
        right_score = float(edges[:, mid:].mean()) if mid else 0.0
        return "left" if left_score <= right_score else "right"

    def _save_evidence(self, frame):
        if frame is None or self.photo_saved:
            return
        now = time.time()
        if now - self.last_photo_time < self.photo_cooldown:
            return
        self.last_photo_time = now
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        label = f"Team 8 detects obstacle on the {self.side} way"
        cv2.putText(annotated, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"IR distance: {self.distance:.0f} cm",
                    (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(annotated, (w // 2, 0), (w // 2, h), (255, 255, 0), 2)
        path = os.path.join(
            self.photo_dir,
            f"obstacle_{self.side}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
        cv2.imwrite(path, annotated)
        self.photo_saved = True
        print(f"Obstacle evidence saved: {path}")

    def _set_state(self, state):
        self.state = state
        self.state_started = time.time()

    def process_detection(self, ep_chassis, frame):
        """Advance the avoidance state machine; return True while it owns chassis."""
        now = time.time()
        elapsed = now - self.state_started

        if self.state == "normal":
            if self.distance >= self.upper_threshold:
                self.armed = True
            if self.armed and self.distance <= self.lower_threshold:
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
                self.side = self._choose_side(frame)
                self.photo_saved = False
                self._save_evidence(frame)
                self.armed = False
                self._set_state("stop")
            return self.state != "normal"

        if self.state == "stop":
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
            if elapsed >= 0.6:
                self._set_state("pass_side")
            return True

        direction = 1 if self.side == "left" else -1
        if self.state == "pass_side":
            ep_chassis.drive_speed(x=0, y=direction * self.lateral_speed,
                                   z=0, timeout=0.2)
            if elapsed >= 1.2:
                self._set_state("pass_forward")
            return True

        if self.state == "pass_forward":
            ep_chassis.drive_speed(x=self.forward_speed, y=0, z=0, timeout=0.2)
            if elapsed >= 1.8:
                self._set_state("return_side")
            return True

        if self.state == "return_side":
            ep_chassis.drive_speed(x=0, y=-direction * self.lateral_speed,
                                   z=0, timeout=0.2)
            if elapsed >= 1.2:
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
                self._set_state("normal")
            return True

        self._set_state("normal")
        return False
