import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import PID
import LowPassFilter


def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
# Simulation parameters
dt = 0.1  # Time step

### Drone Trajectory ###
# Constants
v_d = 25  # speed of the aircraft in m/s
r = 150  # radius of the loiter circle in meters
height = 150 # altitude of the aircraft

# Angular velocity
omega = v_d / r  # rad/s

# Time array
simulation_time = 2 * np.pi * r / v_d  # time to complete one full circle
steps = (int)(simulation_time / dt) + 1

# Initialize arrays to store positions
drone_position = np.zeros((steps,3))

# Initial angle
theta = 0

# Calculate positions
for t in range(0,steps):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    drone_position[t, 0] = x
    drone_position[t, 1] = y
    drone_position[t, 2] = 0
    
    theta += omega * dt

### Target Trajectory ###
v_t = 10
target_position = np.zeros((steps,3))
target_position[:,2] = height
target_position[:,0] = 0 + np.arange(0, steps) * v_t


# Gimbal parameters (roll pitch and yaw)
gimbal_roll_deg = 0.0
gimbal_pitch_deg = -45.0
gimbal_yaw_deg = 180.0

gimbal_roll = np.deg2rad(gimbal_roll_deg)
gimbal_pitch = np.deg2rad(gimbal_pitch_deg)  # Initial pitch angle in radians
gimbal_yaw = np.deg2rad(gimbal_yaw_deg)  # Initial yaw angle in radians
gimbal_pitch_rate = 0.0  # Pitch rotation speed
gimbal_yaw_rate = np.deg2rad(0.0)  # Yaw rotation speed


# PID controller parameters
Kp = 1.0  # Proportional gain
Ki = 0.1 # Integral gain
Kd = 0.0  # Derivative gain

pidh = PID.PID(Kp, Ki, Kd, output_limits=(-30, 30), integral_limits=(-10, 10), derivative_limits=(-10, 10))
pidv = PID.PID(Kp, Ki, Kd, output_limits=(-15, 15), integral_limits=(-5, 5), derivative_limits=(-10, 10))

# PID controller parameters
Kp = 1.0  # Proportional gain
Ki = 0.0 # Integral gain
Kd = 0.0  # Derivative gain

pidh2 = PID.PID(Kp, Ki, Kd, output_limits=(-30, 30), integral_limits=(-15, 15), derivative_limits=(-10, 10))
pidv2 = PID.PID(Kp, Ki, Kd, output_limits=(-30, 30), integral_limits=(-10, 10), derivative_limits=(-10, 10))

# Camera parameters
pixel_pitch = 2.8e-6  # Pixel pitch in meters (2.8 microns)
sensor_size_pixels = np.array([1920, 1080])  # Sensor size in pixels (width, height)
sensor_size_meters = sensor_size_pixels * pixel_pitch  # Sensor size in meters

# Focal length in meters
focal_length = 80e-3  # 129 mm in meters

# Calculate FOV (Field of View)
fov_horizontal = (2 * np.arctan2(sensor_size_meters[0] / 2, focal_length))  # Horizontal FOV in radians
fov_vertical = (2 * np.arctan2(sensor_size_meters[1] / 2, focal_length))  # Vertical FOV in radians

# Camera matrix (Intrinsic parameters)
camera_matrix = np.array([
    [(focal_length / pixel_pitch), 0, sensor_size_pixels[0] / 2],
    [0, (focal_length / pixel_pitch), sensor_size_pixels[1] / 2],
    [0, 0, 1]
])

# No distortion for simplicity
dist_coeffs = np.zeros(4)

# Initialize plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(min(drone_position[:,0]), max(drone_position[:,0]))
ax.set_ylim(min(drone_position[:,1]), max(drone_position[:,1]))
ax.set_zlim(0, height)
ax.set_xlabel('North (meters)')
ax.set_ylabel('East (meters)')
ax.set_zlabel('Down (meters)')
ax.set_title('World View')
ax.invert_zaxis()
ned_map_drone, = ax.plot([], [], [], 'bo', markersize=10)
ned_map_target, = ax.plot([], [], [], 'rx', markersize=10)
ned_map_fov, = ax.plot([], [], [], 'r', markersize=10)


# Create a blank image (Full HD)
image = np.zeros((1080, 1920, 3), dtype=np.uint8)
image.fill(255)

prev_x_pixel = 0
prev_y_pixel = 0
i = 0
ff_yaw = np.deg2rad(9.54) # can be used to initialize feedforward term
ff_pitch = np.deg2rad(0) # can be used to initialize feedforward term
yaw_out = 0
pitch_out = 0
error_yaw_speed = 0
error_pitch_speed = 0

# Initialize arrays to store past yaw and pitch rates for moving average
yaw_rates_history = []
pitch_rates_history = []

ff_yaw_rates_history = []
ff_pitch_rates_history = []

ff_yaw_lowpass = LowPassFilter.LowPassFilter(2.5, dt)
ff_yaw_lowpass.initialize(ff_yaw)
ff_pitch_lowpass = LowPassFilter.LowPassFilter(1, dt)

def moving_average(data, window_size=10):
    """Apply a simple moving average filter."""
    if len(data) < window_size:
        return np.mean(data)  # If not enough data, average what we have
    else:
        return np.mean(data[-window_size:])  # Average the last `window_size` data points

def project_to_camera_opencv(drone_pos, target_pos, pitch, yaw):
    R_roll = Rx(0.0)
    R_pitch = Ry(-pitch)
    R_yaw = Rz(-yaw)

    # World to camera body
    world_to_body = R_roll.dot(R_pitch.dot(R_yaw))

    # Camera body to camera sensor
    body_to_cam = np.matrix([[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]])

    # World to camera sensor
    world_to_cam = body_to_cam.dot(world_to_body)

    # Rotation vector (3x1) for cv2.projectPoints
    rvec, _ = cv2.Rodrigues(world_to_cam)

    # Translation vector (assuming drone position as translation)
    tvec = world_to_cam.dot(-drone_pos[i])

    # Project 3D point to 2D using cv2.projectPoints
    image_points, _ = cv2.projectPoints(np.array([target_pos]), rvec, tvec, camera_matrix, dist_coeffs)

    # Extract pixel coordinates from the result
    x_pixel, y_pixel = image_points[0][0]

    return x_pixel, y_pixel

def update_gimbal_pid(x_pixel, y_pixel):
    global gimbal_pitch, gimbal_yaw
    global gimbal_yaw_rate, gimbal_pitch_rate
    global yaw_out, pitch_out
    global ff_yaw, ff_pitch

    # Calculate the error (difference between target position and center of the frame)
    error_yaw = m.atan2( ((x_pixel - sensor_size_pixels[0] / 2) * pixel_pitch), focal_length )
    error_pitch = -m.atan2( -((y_pixel - sensor_size_pixels[1] / 2) * pixel_pitch ), focal_length )

    if(abs(error_yaw) >= fov_horizontal/2 or abs(error_pitch) >= fov_vertical/2):
       print("Target out of the image. Lost tracking.")
       exit(0)

    _yaw_out = np.deg2rad(pidh.calculate(np.rad2deg(error_yaw), dt))
    _pitch_out = np.deg2rad(pidv.calculate(np.rad2deg(error_pitch), dt))

    yaw_out = np.deg2rad(pidh2.calculate(np.rad2deg(_yaw_out), dt))
    pitch_out = np.deg2rad(pidv2.calculate(np.rad2deg(_pitch_out), dt))

    # Append the latest outputs to the history
    yaw_rates_history.append(yaw_out)
    pitch_rates_history.append(pitch_out)

    # Apply moving average filter to the history to smooth and delay the response
    gimbal_yaw_rate = moving_average(yaw_rates_history, window_size=3) + ff_yaw
    gimbal_pitch_rate = moving_average(pitch_rates_history, window_size=3) + ff_pitch
    
    # Update gimbal angles
    gimbal_yaw += gimbal_yaw_rate * dt
    gimbal_pitch += gimbal_pitch_rate * dt

    #ff_yaw = gimbal_yaw_rate
    #ff_pitch = gimbal_pitch_rate

def update_image(x_pixel, y_pixel):
    # Update tracking box position
    image_ = image.copy()
    cv2.circle(image_, (int(x_pixel), int(y_pixel)), 10, (0, 255, 0), -1)
    cv2.rectangle(image_, (int(x_pixel)-35, int(y_pixel)-35), (int(x_pixel)+35, int(y_pixel)+35), (0, 0, 255), 2)
    
    # Using cv2.putText() method
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    # Draw metadata
    org = (50, 50)
    image_ = cv2.putText(image_, "FoV [" + str(np.rad2deg(fov_horizontal)) + " " + str(np.rad2deg(fov_vertical)) + "] deg", org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (50, 100) 
    image_ = cv2.putText(image_, "Gimbal Rate YPR [" + str(np.rad2deg(gimbal_yaw_rate)) + " " + str(np.rad2deg(gimbal_pitch_rate)) + " 0.]", org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (50, 150) 
    image_ = cv2.putText(image_, "Gimbal FeedForward YPR [" + str(np.rad2deg(ff_yaw)) + " " + str(np.rad2deg(ff_pitch)) + " 0.]", org, font, fontScale, color, thickness, cv2.LINE_AA)  
    org = (50, 200) 
    image_ = cv2.putText(image_, "Gimbal Pose YPR [" + str(np.rad2deg(gimbal_yaw)) + " " + str(np.rad2deg(gimbal_pitch)) + " 0.]", org, font, fontScale, color, thickness, cv2.LINE_AA) 

    cv2.imshow("Camera View", image_)
    cv2.waitKey(1000)

def target_motion_estimator(x_pixel, y_pixel):
    global ff_yaw, ff_pitch
    if( (prev_x_pixel == 0) or (prev_y_pixel == 0) ):
        return

    target_h_speed = (m.atan2( ((x_pixel - sensor_size_pixels[0]) * pixel_pitch), focal_length ) - m.atan2( ((prev_x_pixel - sensor_size_pixels[0]) * pixel_pitch), focal_length ) ) / dt
    target_v_speed = (-m.atan2( -((y_pixel - sensor_size_pixels[1]) * pixel_pitch ), focal_length ) + m.atan2( -((prev_y_pixel - sensor_size_pixels[1]) * pixel_pitch ), focal_length ) ) / dt
    
    target_h_speed_nocomp = target_h_speed + gimbal_yaw_rate
    target_v_speed_nocomp = target_v_speed + gimbal_pitch_rate

    print(f"Target Horizontal Speed w/t compensation: {np.rad2deg(target_h_speed + gimbal_yaw_rate)}")
    print(f"Target Vertical Speed w/t compensation: {np.rad2deg(target_v_speed + gimbal_pitch_rate)}")

    print(f"Target Horizontal Speed: {np.rad2deg(target_h_speed)}")
    print(f"Target Vertical Speed: {np.rad2deg(target_v_speed)}")

    # Append the latest outputs to the history
    ff_yaw_rates_history.append(target_h_speed_nocomp)
    ff_pitch_rates_history.append(target_v_speed_nocomp)

    ff_yaw = ff_yaw_lowpass.filter(target_h_speed_nocomp)
#    ff_yaw = moving_average(ff_yaw_rates_history, window_size=5)
    #ff_pitch = moving_average(ff_pitch_rates_history, window_size=3)

def update(frame):
    global drone_position, target_position, i
    global prev_x_pixel, prev_y_pixel

    # Project target onto the 2D camera plane
    x_pixel, y_pixel = project_to_camera_opencv(drone_position, target_position, gimbal_pitch, gimbal_yaw)

    # Update gimbal angles using PID
    update_gimbal_pid(x_pixel, sensor_size_pixels[1] - y_pixel)

    target_motion_estimator(x_pixel, sensor_size_pixels[1] - y_pixel)

    # Update Camera View image
    update_image(x_pixel, y_pixel)

    #print(f"Drone position: {drone_position[i]}")
    ned_map_drone.set_data_3d(drone_position[i,0],drone_position[i,1], drone_position[i,2])
    ned_map_target.set_data_3d(target_position[i,0],target_position[i,1], target_position[i,2])
    i += 1

    prev_x_pixel = x_pixel
    prev_y_pixel = y_pixel
    return ned_map_drone, ned_map_target,

# Create animation
ani = FuncAnimation(fig, update, frames=int(simulation_time / dt), blit=True, interval=dt*1000)
plt.grid()
plt.show()