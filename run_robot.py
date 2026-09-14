"""总入口：按安全优先级协调所有检测和运动模块。"""

import time

import cv2
from robomaster import robot

from obstacle_avoider import ObstacleAvoider
from line_follower import BlueLineFollower
from marker_detection import MarkerDetector
from traffic_light import TrafficLightDetector


def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False)
    ep_vision = ep_robot.vision
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal

    ep_gimbal.recenter(pitch_speed=180, yaw_speed=180).wait_for_completed()
    ep_gimbal.moveto(pitch=-30, yaw=0, pitch_speed=100, yaw_speed=100).wait_for_completed()
    ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
    ep_robot.led.set_led(comp="all", r=0, g=0, b=0, effect="off")

    marker_detector = MarkerDetector()
    traffic_detector = TrafficLightDetector()
    obstacle_avoider = ObstacleAvoider(ep_robot)
    line_follower = BlueLineFollower()
    ep_vision.sub_detect_info(name="marker", callback=marker_detector.on_detect_marker)

    marker_processing = False
    marker_start_time = 0

    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is None:
                continue

            # 优先级 1：前方机器人。检测到后，其他任务全部暂停。
            if obstacle_avoider.process_detection(ep_chassis, frame.copy()):
                cv2.putText(
                    frame,
                    "Waiting for another robot to leave",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # 优先级 2：红灯停车，绿灯解除等待。
            traffic_detector.process_frame(frame, ep_chassis)

            # 优先级 3：新的数字标记，停车、云台对准、拍照。
            if not marker_processing and not traffic_detector.is_waiting:
                if marker_detector.process(ep_chassis, ep_gimbal, ep_camera, frame.copy()):
                    marker_processing = True
                    marker_start_time = time.time()
                    line_follower.reset_after_interruption()

            if marker_processing:
                if time.time() - marker_start_time >= 3:
                    marker_processing = False
                else:
                    continue

            if traffic_detector.is_waiting:
                continue

            # 优先级 4：正常蓝线循迹；丢线后执行搜索动作。
            mask = line_follower.process(frame, ep_chassis)
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
        ep_vision.unsub_detect_info(name="marker")
        ep_gimbal.drive_speed(0, 0)
        time.sleep(0.5)
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()
        print("Game over")


if __name__ == "__main__":
    main()
