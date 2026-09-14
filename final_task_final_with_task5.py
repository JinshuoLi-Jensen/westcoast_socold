import cv2
from obstacle_avoider import ObstacleAvoider
import numpy as np
from robomaster import robot
from robomaster import camera
import time
import threading
import os
import math
from datetime import datetime


# 增强版PID控制器类
class EnhancedPID:
    def __init__(self, p=1.1, i=0.1, d=0.05, out_limit=80, integral_limit=50):
        self.kp = p
        self.ki = i
        self.kd = d
        self.out_limit = out_limit
        self.integral_limit = integral_limit
        self.clear()

    def clear(self):
        self.set_point = 0.0
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, target, current):
        # 计算时间差
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # 避免除零错误
        if dt <= 0:
            dt = 0.01

        error = target - current

        # 比例项
        p_term = self.kp * error

        # 积分项（带积分限幅）
        self.integral += error * dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_term = self.ki * self.integral

        # 微分项
        d_term = self.kd * (error - self.last_error) / dt
        self.last_error = error

        # 总输出
        output = p_term + i_term + d_term

        # 输出限制
        if output > self.out_limit:
            output = self.out_limit
        elif output < -self.out_limit:
            output = -self.out_limit

        return output

class MarkerInfo:
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

detected_targets = set()
markers = []
markers_lock = threading.Lock()
photo_taken_targets = set()

def on_detect_marker(marker_info):
    global markers
    number = len(marker_info)
    with markers_lock:
        markers.clear()
        for i in range(0, number):
            x, y, w, h, info = marker_info[i]
            markers.append(MarkerInfo(x, y, w, h, info))

def track_marker_and_take_photo(ep_gimbal, ep_camera, save_dir, target_text, center_threshold=5):  # 修改：增加target_text参数
    # p控制
    kp_yaw = 20.0
    kp_pitch = 30.5
    kp_gimbal_yaw = 20.0

    ref_w = 0.3
    kp_distance = 2.8
    distance_deadzone = 0.002

    # 死区
    deadzone_x = 0.003
    deadzone_y = 0.003

    has_taken_photo = False

    start_time = time.time()
    timeout = 10

    while time.time() - start_time < timeout:
        # 获取最新相机帧
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if img is None:
            continue

        # 图像中心点坐标，使线保持在小车中心
        img_center = (640, 360)

        with markers_lock:
            target_marker = None
            # 查找目标标签
            for j in range(len(markers)):
                if markers[j].text == target_text:
                    target_marker = markers[j]
                    break

            # 找到目标标签
            if target_marker is not None:
                # 绘制标签边框和信息
                cv2.rectangle(img, target_marker.pt1, target_marker.pt2, (255, 255, 255))
                cv2.putText(img, target_marker.text, target_marker.center,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                # 绘制标签中心点
                marker_center = target_marker.center
                cv2.circle(img, marker_center, 10, (0, 0, 255), -1)
                cv2.circle(img, marker_center, 12, (255, 0, 0), 2)

                # 标签跟踪控制
                marker_x = target_marker.x
                marker_y = target_marker.y
                marker_w = target_marker.w

                # 计算位置误差
                error_x = marker_x - 0.5
                error_y = marker_y - 0.5
                # 计算距离误差
                error_distance = ref_w - marker_w

                # 应用死区
                error_x = 0 if abs(error_x) < deadzone_x else error_x
                error_y = 0 if abs(error_y) < deadzone_y else error_y
                error_distance = 0 if abs(error_distance) < distance_deadzone else error_distance

                # 计算控制速度
                z_speed = -kp_yaw * error_x
                pitch_speed = -kp_pitch * error_y
                gimbal_yaw_speed = -kp_gimbal_yaw * z_speed

                # 限制速度
                pitch_speed = max(-60, min(60, pitch_speed))
                gimbal_yaw_speed = max(-60, min(60, gimbal_yaw_speed))

                # 发送云台控制指令
                ep_gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=gimbal_yaw_speed)

                # 中心点重合判断与拍照
                distance = ((marker_center[0] - img_center[0]) ** 2 +
                            (marker_center[1] - img_center[1]) ** 2) ** 0.5

                # 满足重合且未拍照的条件时触发拍照
                if distance < center_threshold and not has_taken_photo:
                    cv2.line(img, (640, 0), (640, 720), (0, 255, 0), 2)
                    cv2.line(img, (0, 360), (1280, 360), (0, 255, 0), 2)

                    photo_name = f"marker_photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    photo_path = os.path.join(save_dir, photo_name)

                    # 保存图像
                    if img is not None:
                        # 添加文字
                        text = f"Team 8 detects marker with ID of {target_marker.text}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.5
                        font_color = (0, 255, 255)
                        thickness = 3
                        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                        text_x = (img.shape[1] - text_size[0]) // 2
                        text_y = 50
                        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness)

                        cv2.imwrite(photo_path, img)
                        print(f"Photo has taken！Save photo to:{photo_path}")
                        has_taken_photo = True

                        # 低头
                        ep_gimbal.moveto(pitch=-30, yaw=0, pitch_speed=100, yaw_speed=100).wait_for_completed()

                        # 短暂延迟确保动作完成
                        time.sleep(1)

                        return True

                # 中心点不重合时，重置标志位
                elif distance >= center_threshold:
                    has_taken_photo = False
            else:
                # 没有找到目标标签：停止云台运动
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                cv2.putText(img, f"Target marker {target_text} not found", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 已经拍照，退出循环
        if has_taken_photo:
            break

        # 绘制图像中心点十字线
        cv2.line(img, (640, 0), (640, 720), (0, 255, 0), 2)  # 垂直中线
        cv2.line(img, (0, 360), (1280, 360), (0, 255, 0), 2)  # 水平中线

        # 显示处理后的图像
        cv2.imshow("Marker Tracking", img)

        # 按下「q」键退出循环
        if cv2.waitKey(1) == ord('q'):
            break

    # 超时或用户中断，返回False
    return False

def process_detected_markers(ep_chassis, ep_gimbal, ep_camera, image):
    global detected_targets, photo_taken_targets
    current_target = None

    with markers_lock:
        for marker in markers:
            # 绘制所有检测到的标记
            cv2.rectangle(image, marker.pt1, marker.pt2, (255, 255, 255))
            cv2.putText(image, marker.text, marker.center, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # 检查是否是目标标签(1-5)且未被识别过，并且标记足够大
            if (marker.text in ['1', '2', '3', '4', '5'] and
                    marker.text not in detected_targets and
                    marker.text not in photo_taken_targets and
                    marker.w > 0.1 and marker.h > 0.1):

                current_target = marker.text
                detected_targets.add(marker.text)

    # 如果检测到目标标签且未被识别过，则停车并显示
    if current_target:
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        time.sleep(1)
        print(f"Group 8, detect target marker: {current_target}，Stop!")

        # 设置拍照保存目录
        save_dir = r"D:\robotest"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 调用云台跟踪和拍照函数
        photo_taken = track_marker_and_take_photo(ep_gimbal, ep_camera, save_dir, current_target)

        if photo_taken:
            # 将该目标添加到已拍照集合中
            photo_taken_targets.add(current_target)
            # 云台回正
            ep_gimbal.moveto(pitch=-30, yaw=0, pitch_speed=100, yaw_speed=100).wait_for_completed()
            time.sleep(1)
        else:
            print("Photo timeout or failure, continue patrolling the line")

        return True

    return False

# 红绿灯检测功能
class TrafficLightDetector:
    def __init__(self):
        self.is_waiting = False
        self.last_detection_time = 0
        self.cooldown = 0.5
        self.image_count = 0
        self.last_light_color = None

        # 图像尺寸
        self.img_width = 640
        self.img_height = 480

        # HSV范围
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([8, 255, 255])
        self.red_lower2 = np.array([172, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        self.green_lower = np.array([50, 80, 80])
        self.green_upper = np.array([70, 255, 255])

        # 最小检测面积
        self.min_area = 100

        # 圆形度阈值
        self.circularity_threshold = 0.79

        # 最小半径
        self.min_radius = 12

        print("The traffic light detector is initialized.")

    #圆形度计算
    def calculate_circularity(self, contour):
        area = cv2.contourArea(contour)
        if area == 0:
            return 0

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        return circularity

    #半径计算
    def calculate_radius(self, contour):
        area = cv2.contourArea(contour)
        if area == 0:
            return 0
        return math.sqrt(area / math.pi)

    #红绿灯检测
    def detect_light(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 检测红色
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # 检测绿色
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        detected_color = None
        best_contour = None
        max_circularity = 0

        # 检查红色区域
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:  #检查面积
                # 检查半径
                radius = self.calculate_radius(contour)
                if radius < self.min_radius:
                    continue

                circularity = self.calculate_circularity(contour)
                if circularity > self.circularity_threshold and circularity > max_circularity:  #检查圆形度
                    max_circularity = circularity
                    detected_color = 'red'
                    best_contour = contour

        # 检查绿色区域
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in green_contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # 检查半径
                radius = self.calculate_radius(contour)
                if radius < self.min_radius:
                    continue

                circularity = self.calculate_circularity(contour)
                if circularity > self.circularity_threshold and circularity > max_circularity:
                    max_circularity = circularity
                    detected_color = 'green'
                    best_contour = contour

        return detected_color, best_contour

    #保存照片
    def save_detection_image(self, frame, color, contour):
        if contour is None:
            return
        if color == self.last_light_color:
            return

        # 绘制边框并保存
        annotated_frame = frame.copy()
        cv2.drawContours(annotated_frame, [contour], -1, (0, 255, 255), 2)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"traffic_light_{color}_{timestamp}_{self.image_count:04d}.jpg"
        save_dir = r"D:\robotest"
        photo_path = os.path.join(save_dir, filename)

        # 添加文字
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            # 文字内容
            if color == 'red':
                label = "Team 8 detects a red light"
            else:
                label = "Team 8 detects a green light"
            # 字体和大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # 文字尺寸
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # 文字位置
            text_x = x + (w - text_width) // 2
            text_y = y + h + text_height + 5

            bg_color = (0, 0, 0)
            alpha = 0.6
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay,
                          (text_x - 2, text_y - text_height - 2),
                          (text_x + text_width + 2, text_y + 2),
                          bg_color, -1)
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

            # 绘制文字
            text_color = (0, 255, 255)
            cv2.putText(annotated_frame, label, (text_x, text_y),
                        font, font_scale, text_color, thickness)

        cv2.imwrite(photo_path, annotated_frame)

        self.image_count += 1
        print(f"saved image: {filename}")
        self.last_light_color = color

    def process_frame(self, frame, ep_chassis):
        if frame is None:
            return

        frame = cv2.resize(frame, (self.img_width, self.img_height))

        current_time = time.time()
        # 检测时间间隔
        if current_time - self.last_detection_time < self.cooldown:
            return

        # 检测红绿灯
        light_color, contour = self.detect_light(frame)
        self.last_detection_time = current_time

        # 保存图像
        if light_color:
            self.save_detection_image(frame, light_color, contour)

        # 控制机器人运动
        if light_color == 'red':
            ep_chassis.drive_speed(x=0, y=0, z=0)
            self.is_waiting = True
            print("red light detected, stop moving")

        elif light_color == 'green':
            if self.is_waiting:
                self.is_waiting = False
                print("green light detected, moving")

class InfraredDetector:
    def __init__(self, ep_robot):
        # 初始化机器人组件
        self.ep_robot = ep_robot
        self.ep_sensor = ep_robot.sensor

        # 红外传感器阈值
        self.ir_upper_threshold = 500
        self.ir_lower_threshold = 340

        # 状态变量
        self.detected_condition = False
        self.last_detected_time = 0
        self.last_photo_time = 0
        self.photo_cooldown = 10.0
        self.has_taken_photo_for_current_obstacle = False
        self.fully_stopped = False
        self.stop_start_time = 0

        # 红外传感器数据
        self.ir_distance = 0

        # 初始化传感器订阅
        self.ep_sensor.sub_distance(freq=5, callback=self.ir_sensor_callback)

        # 创建照片保存目录
        self.photo_dir = r"D:\robotest"
        if not os.path.exists(self.photo_dir):
            os.makedirs(self.photo_dir)

    def ir_sensor_callback(self, sub_info):
        self.ir_distance = sub_info[0]

    def check_ir_sensor(self):
        return self.ir_distance

    def update_detection_state(self):
        current_time = time.time()
        # 检查红外传感器
        if self.check_ir_sensor() <= self.ir_lower_threshold:
            self.detected_condition = True
            self.last_detected_time = current_time
            return True
        elif self.check_ir_sensor() >= self.ir_upper_threshold:
            # 机器人存在但距离不够近，不停止但记录检测时间
            self.detected_condition = False
            self.last_detected_time = current_time
            return False
        else:
            # 保持之前的状态
            return self.detected_condition

    def take_photo(self, frame):
        current_time = time.time()

        # 检查冷却时间
        if current_time - self.last_photo_time < self.photo_cooldown:
            return False

        # 更新拍照时间
        self.last_photo_time = current_time

        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 添加红外距离信息
        cv2.putText(frame, f"IR: {self.ir_distance}cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 保存照片
        filename = os.path.join(self.photo_dir, f"robot_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        text = f"Team 8 detected a robot"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_color = (0, 255, 255)
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 50  # 距离顶部50像素
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

        height, width = frame.shape[:2]
        cv2.rectangle(frame, (90, 70), (width - 90, height - 70), (0, 0, 255), 10)

        cv2.imwrite(filename, frame)
        print(f"已保存机器人检测照片: {filename}")
        return True

    def process_detection(self, ep_chassis, frame):
        current_time = time.time()
        ir_value = self.check_ir_sensor()

        # 检测障碍物是否出现
        if ir_value <= self.ir_lower_threshold:
            # 立即停止机器人
            ep_chassis.drive_speed(x=0, y=0, z=0)

            # 如果是新检测到的障碍物，记录停止时间
            if not self.detected_condition:
                self.detected_condition = True
                self.fully_stopped = False
                self.stop_start_time = current_time
                print("检测到另一个机器人，开始停止")

            # 检查是否已经完全停止
            if not self.fully_stopped and current_time - self.stop_start_time > 0.5:
                self.fully_stopped = True
                print("机器人已完全停止")

            # 如果已经完全停止且未拍照，则拍照
            if self.fully_stopped and not self.has_taken_photo_for_current_obstacle:
                self.take_photo(frame)
                self.has_taken_photo_for_current_obstacle = True
                print("已拍摄另一个机器人的照片")

            return True

        # 障碍物离开
        elif ir_value >= self.ir_upper_threshold:
            self.has_taken_photo_for_current_obstacle = False
            self.detected_condition = False
            self.fully_stopped = False
            return False

        else:
            # 距离在上下限之间，保持之前的状态
            if self.detected_condition:
                ep_chassis.drive_speed(x=0, y=0, z=0)
            return self.detected_condition

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

    ep_led = ep_robot.led
    ep_led.set_led(comp='all', r=0, g=0, b=0, effect='off')

    # 初始化标记检测
    result = ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    # 初始化PID控制器
    pid_controller = EnhancedPID(p=0.4, i=0.001, d=0.05, out_limit=80)

    # 初始化红绿灯检测器
    traffic_light_detector = TrafficLightDetector()

    # 初始化红外检测器
    # Task 5: stop, choose a clear side, pass the obstacle and return to the line.
    infrared_detector = ObstacleAvoider(ep_robot)

    # 蓝色HSV范围
    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([130, 255, 255])

    # 动态获取图像尺寸
    frame_width = None
    frame_center = None

    # 状态变量
    line_lost_count = 0
    max_line_lost = 10

    # 新增搜索状态变量
    search_state = "idle"
    search_start_time = 0

    # 历史位置记录
    position_history = []
    history_length = 8

    # 标记检测状态
    marker_detected = False
    marker_processing = False
    marker_start_time = 0

    try:
        while True:
            # 获取图像帧
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)

            if frame is None:
                continue

            # 如果是第一次获取帧，初始化宽度和中心点
            if frame_width is None:
                height, frame_width = frame.shape[:2]
                frame_center = frame_width // 2
                print(f"检测到图像尺寸: 宽度={frame_width}, 高度={height}")
                print(f"图像中心点: {frame_center}")

            # 处理红外传感器检测
            infrared_detected = infrared_detector.process_detection(ep_chassis, frame.copy())

            # 如果检测到另一个机器人，跳过其他处理
            if infrared_detected:
                # 显示等待状态
                cv2.putText(frame, "Waiting for another robot to leave", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
                continue

            # 处理红绿灯检测
            traffic_light_detector.process_frame(frame, ep_chassis)

            # 处理标记检测
            if not marker_processing and not traffic_light_detector.is_waiting:
                target_detected = process_detected_markers(ep_chassis, ep_gimbal, ep_camera, frame.copy())
                if target_detected:
                    marker_processing = True
                    marker_start_time = time.time()
                    # 重置PID控制器
                    pid_controller.clear()
                    # 重置历史位置
                    position_history.clear()

            # 如果正在处理标记，检查是否已经处理了3秒
            if marker_processing:
                # 如果已经过了3秒，则结束处理标记状态
                if time.time() - marker_start_time >= 3:
                    marker_processing = False
                else:
                    # 否则，跳过循线，继续等待
                    continue

            # 如果正在等待绿灯，跳过循线
            if traffic_light_detector.is_waiting:
                continue

            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 创建蓝色掩膜
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 高斯模糊减少噪声
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # 选择多个ROI行来提高稳定性
            height, width = mask.shape
            roi_rows = [height - 400, height - 300, height - 250, height - 200, height - 100, height - 70, height - 40,
                        height - 20]

            line_centers = []

            for roi_row in roi_rows:
                # 提取该行的所有像素
                roi = mask[roi_row, :]

                # 找到白色像素的位置
                white_pixels = np.where(roi == 255)[0]

                if len(white_pixels) > 10:
                    # 计算白色像素的中心点
                    line_center = int(np.mean(white_pixels))
                    line_centers.append(line_center)

                    # 在图像上绘制参考线和中点
                    cv2.line(frame, (0, roi_row), (width, roi_row), (0, 255, 0), 1)
                    cv2.circle(frame, (line_center, roi_row), 5, (0, 0, 255), -1)

            # 计算平均中心点
            if line_centers:
                # 重置搜索状态
                search_state = "idle"

                # 重置丢失计数器
                line_lost_count = 0

                # 使用历史数据平滑处理
                avg_line_center = int(np.mean(line_centers))
                position_history.append(avg_line_center)
                if len(position_history) > history_length:
                    position_history.pop(0)

                smoothed_center = int(np.mean(position_history))

                # 计算与图像中心的偏差
                error = frame_center - smoothed_center

                # 使用PID控制器计算转向速度
                z_speed = pid_controller.update(0, error)

                # 根据偏差大小调整前进速度
                base_speed = 0.5
                speed_reduction = min(abs(error) / 100.0, 0.2)
                x_speed = base_speed - speed_reduction

                # 只控制底盘，不控制云台
                ep_chassis.drive_speed(x=x_speed, y=0, z=z_speed, timeout=0.1)

                # 在图像上绘制中心点
                cv2.circle(frame, (smoothed_center, height - 50), 8, (255, 0, 0), -1)
                cv2.circle(frame, (frame_center, height - 50), 8, (0, 255, 255), -1)

                # 显示调试信息
                cv2.putText(frame, f"Error: {error}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Z-Speed: {z_speed:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            else:
                # 没有检测到蓝线
                line_lost_count += 1
                if line_lost_count > max_line_lost:

                    # 进入寻找蓝线模式
                    current_time = time.time()
                    if search_state == "idle":
                        # 初始状态，开始第一次左转
                        search_state = "left_turn_first"
                        search_start_time = current_time
                        print("开始")

                    # 第一次左转90度
                    elif search_state == "left_turn_first":
                        # 旋转90度
                        ep_chassis.drive_speed(x=0, y=0, z=30, timeout=0.1)
                        cv2.putText(frame, "SEARCHING: Turning left 90°", (width // 2 - 150, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # 检查是否完成左转
                        if current_time - search_start_time > 2.0:
                            search_state = "check_after_left"
                            search_start_time = current_time
                            print("第一次左转完成")

                    # 左转后停止检查
                    elif search_state == "check_after_left":
                        # 停止所有运动，给时间检查
                        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                        cv2.putText(frame, "SEARCHING: Checking after left turn", (width // 2 - 180, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        # 停留1秒用于检查
                        if current_time - search_start_time > 1.0:
                            search_state = "right_turn_first"
                            search_start_time = current_time
                            print("检查完成")

                    # 第一次右转90度
                    elif search_state == "right_turn_first":
                        # 旋转-90度（右转）
                        ep_chassis.drive_speed(x=0, y=0, z=-30, timeout=0.1)
                        cv2.putText(frame, "SEARCHING: Turning right 90° (1/2)", (width // 2 - 180, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # 检查是否完成右转
                        if current_time - search_start_time > 2.0:
                            search_state = "right_turn_second"
                            search_start_time = current_time
                            print("第一次右转完成")

                    # 第二次右转90度
                    elif search_state == "right_turn_second":
                        # 继续旋转-90度
                        ep_chassis.drive_speed(x=0, y=0, z=-30, timeout=0.1)
                        cv2.putText(frame, "SEARCHING: Turning right 90° (2/2)", (width // 2 - 180, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # 检查是否完成第二次右转
                        if current_time - search_start_time > 2.0:
                            search_state = "check_after_rights"
                            search_start_time = current_time
                            print("第二次右转完成")

                    # 两次右转后停止检查
                    elif search_state == "check_after_rights":
                        # 停止所有运动，给时间检查
                        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                        cv2.putText(frame, "SEARCHING: Checking after right turns", (width // 2 - 190, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        # 停留1秒用于检查
                        if current_time - search_start_time > 1.0:
                            search_state = "left_turn_final"
                            search_start_time = current_time
                            print("检查完成")

                    # 6. 最后左转90度
                    elif search_state == "left_turn_final":
                        # 旋转90度回到初始朝向
                        ep_chassis.drive_speed(x=0, y=0, z=30, timeout=0.1)
                        cv2.putText(frame, "SEARCHING: Final left turn 90°", (width // 2 - 180, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # 检查是否完成最后左转
                        if current_time - search_start_time > 2.0:
                            search_state = "move_forward"
                            search_start_time = current_time
                            print("最后左转完成，准备前进0.1米")

                    # 7. 前进0.1米
                    elif search_state == "move_forward":
                        ep_chassis.drive_speed(x=0.2, y=0, z=0, timeout=0.1)
                        cv2.putText(frame, "SEARCHING: Moving forward 0.1m", (width // 2 - 170, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # 检查是否完成前进
                        if current_time - search_start_time > 1:
                            search_state = "left_turn_first"
                            search_start_time = current_time
                            print("前进完成")

                else:
                    pass

            # 显示图像
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 停止机器人并清理资源
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
        result = ep_vision.unsub_detect_info(name="marker")
        ep_gimbal.drive_speed(0, 0)
        time.sleep(0.5)
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()
        print("Game over")

if __name__ == "__main__":
    main()
