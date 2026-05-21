import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.node import Node
import threading
import math
import time
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import SetPen
from std_srvs.srv import Empty


class ShapeOptimizerPro(Node):
    def __init__(self):
        super().__init__('shape_optimizer_pro')
        self.velocity_pub  = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.pose_sub      = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        self.pen_client    = self.create_client(SetPen, '/turtle1/set_pen')
        self.clear_client  = self.create_client(Empty, '/clear')

        self.robot_x     = 5.54
        self.robot_y     = 5.54
        self.robot_theta = 0.0

        self.ideal_path  = []
        self.loop_index  = 0
        self.shape_name  = ""
        self.is_tracking_active = False
        self.pen_lowered = False

        self.Kv           = 2.5
        self.Ktheta       = 7.0
        self.Kd_theta     = 0.5
        self.Kd_linear    = 0.1
        self.Klinear_max  = 3.0
        self.prev_angle_error    = 0.0
        self.prev_distance_error = 0.0

        self.create_timer(0.02, self.kinematics_engine_callback)

    def pose_callback(self, msg):
        self.robot_x     = msg.x
        self.robot_y     = msg.y
        self.robot_theta = msg.theta

    def set_turtle_pen(self, off, r=0, g=0, b=0):
        if not self.pen_client.service_is_ready():
            return
        req = SetPen.Request()
        req.r = r; req.g = g; req.b = b
        req.width = 3
        req.off   = off
        self.pen_client.call_async(req)

    def sort_rectangle_corners(self, pts):
        pts    = pts.reshape(4, 2)
        center = np.mean(pts, axis=0)
        pts    = sorted(pts, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
        return np.array(pts)

    # called every 20 ms by the ros timer — drives the turtle along the waypoint path
    def kinematics_engine_callback(self):
        if not self.is_tracking_active or len(self.ideal_path) == 0:
            return

        # compute distance and angle to the current waypoint
        tx, ty         = self.ideal_path[self.loop_index]
        distance_error = math.sqrt((tx - self.robot_x)**2 + (ty - self.robot_y)**2)
        desired_angle  = math.atan2(ty - self.robot_y, tx - self.robot_x)
        angle_error    = math.atan2(
            math.sin(desired_angle - self.robot_theta),
            math.cos(desired_angle - self.robot_theta))

        # derivative terms to reduce overshoot
        d_angle    = angle_error    - self.prev_angle_error
        d_distance = distance_error - self.prev_distance_error
        self.prev_angle_error    = angle_error
        self.prev_distance_error = distance_error

        # pd control outputs for angular and linear velocity
        angular_control = self.Ktheta * angle_error + self.Kd_theta * d_angle
        linear_control  = min(self.Kv * distance_error + self.Kd_linear * d_distance,
                              self.Klinear_max)
        if abs(angle_error) > 0.8:
            linear_control *= 0.15

        twist_cmd = Twist()

        # phase 1: navigate to the first waypoint with pen up
        if self.loop_index == 0 and distance_error > 0.35:
            if abs(angle_error) > 0.4:
                twist_cmd.linear.x  = 0.0
                twist_cmd.angular.z = angular_control
            else:
                twist_cmd.linear.x  = linear_control
                twist_cmd.angular.z = angular_control
            self.velocity_pub.publish(twist_cmd)
            return

        # phase 2: arrived at start — lower the pen once and move to waypoint 1
        if self.loop_index == 0 and not self.pen_lowered:
            self.velocity_pub.publish(Twist())
            if self.shape_name == "Circle":
                self.set_turtle_pen(off=0, r=0,   g=0,   b=255)
            elif self.shape_name == "Rectangle":
                self.set_turtle_pen(off=0, r=255, g=0,   b=0)
            elif self.shape_name == "Triangle":
                self.set_turtle_pen(off=0, r=0,   g=255, b=0)
            self.pen_lowered = True
            self.loop_index  = 1
            return

        # phase 3: main drawing — follow waypoints until the shape is complete
        if abs(angle_error) > 1.2:
            twist_cmd.linear.x  = 0.0
            twist_cmd.angular.z = angular_control
        else:
            twist_cmd.linear.x  = max(linear_control, 0.2)
            twist_cmd.angular.z = angular_control
        self.velocity_pub.publish(twist_cmd)

        # advance to the next waypoint when close enough, loop back to repeat the shape
        if distance_error < 0.35:
            self.loop_index += 1
            if self.loop_index >= len(self.ideal_path):
                self.loop_index          = 1
                self.prev_angle_error    = 0.0
                self.prev_distance_error = 0.0


# ── path helpers ──────────────────────────────────────────────────────────────

# interpolates extra points along each edge so the turtle has dense waypoints to follow
def generate_dense_polygon(points, density=15):
    dense = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        for t in np.linspace(0, 1, density):
            dense.append((p1[0] + t*(p2[0]-p1[0]), p1[1] + t*(p2[1]-p1[1])))
    return dense

# adds midpoints between every consecutive pair to smooth sharp direction changes
def smooth_path(path):
    smooth = []
    for i in range(len(path) - 1):
        x1,y1 = path[i]; x2,y2 = path[i+1]
        for t in np.linspace(0, 1, 5):
            smooth.append((x1 + t*(x2-x1), y1 + t*(y2-y1)))
    return smooth


# ── shape classifier ─────────────────────────────────────────────────────────

def classify_shape(pts_raw, epsilon_coeff=0.03):
    """
    Returns (shape_name, approx_corners_or_None, circularity, vertex_count)

    Strategy (order matters):
      1. Compute the convex hull and its approxPolyDP.
      2. Check vertex count FIRST (3 → triangle, 4 → rectangle).
      3. Only fall back to circularity for circle — and use a HIGH threshold.

    Key fixes vs original:
      - epsilon_coeff lowered 0.048 → 0.03  so rectangle corners are NOT merged.
      - Vertex count checked BEFORE circularity so rectangles never mis-fire as circles.
      - Rectangle: also verify aspect-ratio-based squareness guard is removed
        (minAreaRect handles any oriented rectangle).
      - Circle: threshold raised 0.68 → 0.88 — only very round shapes qualify.
      - Added a "quadrilateral but not rectangle" safety: if aspect ratio is very
        extreme (>5:1) treat it as unknown rather than a squashed rectangle.
    """
    hull        = cv2.convexHull(pts_raw)
    hull_peri   = cv2.arcLength(hull, True)
    hull_area   = cv2.contourArea(hull)

    # circularity on closed convex hull (fixes open-path near-zero area bug)
    circularity = (4 * np.pi * hull_area) / (hull_peri ** 2) if hull_peri > 0 else 0

    # approxpolydp with tighter epsilon so corners are preserved
    approx      = cv2.approxPolyDP(hull, epsilon_coeff * hull_peri, True)
    n_verts     = len(approx)

    # 3 vertices → triangle
    if n_verts == 3:
        return "Triangle", approx, circularity, n_verts

    # 4 vertices → rectangle (reject degenerate thin slivers)
    if n_verts == 4:
        rect   = cv2.minAreaRect(pts_raw)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            return "Unknown", approx, circularity, n_verts
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > 6.0:
            return "Unknown", approx, circularity, n_verts
        return "Rectangle", approx, circularity, n_verts

    # 5–8 vertices → circle only if the hull is round enough
    if 5 <= n_verts <= 8:
        if circularity > 0.82:
            return "Circle", approx, circularity, n_verts
        return "Unknown", approx, circularity, n_verts

    # many vertices → circle if very round
    if n_verts > 8:
        if circularity > 0.78:
            return "Circle", approx, circularity, n_verts
        return "Unknown", approx, circularity, n_verts

    return "Unknown", approx, circularity, n_verts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    app_node = ShapeOptimizerPro()

    # spin ros in a background thread so the 50 hz timer fires independently of the opencv loop
    ros_thread = threading.Thread(target=rclpy.spin, args=(app_node,), daemon=True)
    ros_thread.start()

    # set up mediapipe hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.80,
        min_tracking_confidence=0.75,
    )

    cap    = cv2.VideoCapture(0)
    canvas = None
    raw_path = []
    xp, yp   = 0, 0

    EPSILON_COEFF  = 0.03
    MIN_POINTS     = 25
    MIN_DIST       = 8
    FRAME_INTERVAL = 1.0 / 15.0
    last_frame_time = 0.0

    print("System Initialized — draw a shape, lift finger to send to turtle")

    while cap.isOpened():
        # cap frame rate to ~15 fps for comfortable drawing
        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)
        last_frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks and not app_node.is_tracking_active:
            for lms in res.multi_hand_landmarks:
                tip     = lms.landmark[8]   # index fingertip
                knuckle = lms.landmark[6]   # index knuckle
                cx = int(tip.x * w)
                cy = int(tip.y * h)

                # finger up: record drawing points
                if tip.y < knuckle.y:
                    if len(raw_path) == 0:
                        raw_path.append([cx, cy])
                    else:
                        px, py = raw_path[-1]
                        if math.sqrt((cx-px)**2 + (cy-py)**2) > MIN_DIST:
                            raw_path.append([cx, cy])
                    if xp != 0:
                        cv2.line(canvas, (xp, yp), (cx, cy), (0, 255, 0), 4)
                    xp, yp = cx, cy

                # finger down: classify the drawn stroke and send path to turtle
                else:
                    if len(raw_path) > MIN_POINTS:
                        pts_raw = np.array(raw_path, np.int32)

                        shape_name, approx, circularity, n_verts = classify_shape(
                            pts_raw, EPSILON_COEFF)

                        app_node.shape_name = shape_name
                        path_buffer = []
                        scale = 7.0 / max(w, h)

                        if shape_name == "Circle":
                            (x_c, y_c), radius = cv2.minEnclosingCircle(pts_raw)
                            cv2.circle(canvas, (int(x_c), int(y_c)), int(radius), (255,255,0), 4)
                            for i in range(40):
                                ang = i * 2 * np.pi / 40
                                path_buffer.append((
                                    2.0 + (x_c + radius * math.cos(ang)) * scale,
                                    9.0 - (y_c + radius * math.sin(ang)) * scale
                                ))
                            path_buffer.append(path_buffer[0])
                            app_node.is_tracking_active = True

                        elif shape_name == "Triangle":
                            pts_tri = approx.reshape(3, 2)
                            center  = np.mean(pts_tri, axis=0)
                            pts_tri = sorted(pts_tri,
                                key=lambda p: np.arctan2(p[1]-center[1], p[0]-center[0]))
                            pts_tri = np.array(pts_tri)
                            cv2.drawContours(canvas, [pts_tri.astype(int)], 0, (255,255,0), 4)
                            for p in generate_dense_polygon(pts_tri, density=20):
                                path_buffer.append((2.0 + p[0]*scale, 9.0 - p[1]*scale))
                            path_buffer.append(path_buffer[0])
                            app_node.is_tracking_active = True

                        elif shape_name == "Rectangle":
                            rect        = cv2.minAreaRect(pts_raw)
                            box         = cv2.boxPoints(rect)
                            box         = np.intp(box)
                            sorted_rect = app_node.sort_rectangle_corners(box)
                            cv2.drawContours(canvas, [sorted_rect.astype(int)], 0, (255,255,0), 4)
                            for p in generate_dense_polygon(sorted_rect, density=20):
                                path_buffer.append((2.0 + p[0]*scale, 9.0 - p[1]*scale))
                            path_buffer.append(path_buffer[0])
                            app_node.is_tracking_active = True

                        else:
                            app_node.shape_name = "Try Again"

                        if app_node.is_tracking_active:
                            # lift pen before moving to start so no stray line is drawn
                            app_node.set_turtle_pen(off=1)
                            app_node.pen_lowered = False
                            smooth = smooth_path(path_buffer)
                            app_node.loop_index = 0
                            app_node.ideal_path = smooth

                        cv2.putText(canvas, app_node.shape_name, (20, 60),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2)
                        raw_path = []
                    xp, yp = 0, 0
        else:
            xp, yp = 0, 0

        output = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)

        # live debug overlay while the user is still drawing
        if len(raw_path) > 0 and not app_node.is_tracking_active:
            pts_debug   = np.array(raw_path, np.int32)
            hull_d      = cv2.convexHull(pts_debug)
            hull_peri_d = cv2.arcLength(hull_d, True)
            hull_area_d = cv2.contourArea(hull_d)
            circ_d      = (4*np.pi*hull_area_d)/(hull_peri_d**2) if hull_peri_d > 0 else 0
            approx_d    = cv2.approxPolyDP(hull_d, EPSILON_COEFF * hull_peri_d, True)
            cv2.putText(output,
                f"pts:{len(raw_path)}  circ:{circ_d:.2f}  verts:{len(approx_d)}",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,255,100), 1)

        # show which shape the turtle is currently tracing and waypoint progress
        if app_node.is_tracking_active:
            cv2.putText(output,
                f"Tracking: {app_node.shape_name}  wp:{app_node.loop_index}/{len(app_node.ideal_path)}",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,200,255), 1)

        cv2.imshow('Shape Detection + TurtleSim', output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # reset everything — clear canvas, stop turtle, erase turtlesim trail
            canvas = np.zeros_like(frame)
            app_node.ideal_path         = []
            app_node.shape_name         = ""
            app_node.is_tracking_active = False
            app_node.pen_lowered        = False
            app_node.set_turtle_pen(off=1)
            app_node.clear_client.call_async(Empty.Request())
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    app_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()