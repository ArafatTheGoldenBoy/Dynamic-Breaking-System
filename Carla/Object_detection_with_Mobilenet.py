from __future__ import print_function

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            '../carla/dist/carla-*%d.%d-%s.egg' % (
                sys.version_info.major,
                sys.version_info.minor,
                'win-amd64' if os.name=='nt' else 'linux-x86_64')
        )[0]
    )
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla
import pygame
import cv2
import numpy as np
import random
import math
import time
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ESCAPE, K_a, K_q

# Global variables to store the latest camera image in two formats:
image_surface = None   # pygame Surface for the gameplay window
opencv_frame = None    # OpenCV BGR image for detection

# ==============================================================================
# -- Load DNN Model for Object Detection --------------------------------------
# ==============================================================================
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# Create the DNN detection model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels from file
classLabels = []
with open('labels.txt', 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')

# Configure the model (tune these parameters as needed)
model.setInputSize(300, 300)   # Larger input size may give better results but is slower
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# For FPS computation in the detection pipeline
prev_frame_time = time.time()

# ==============================================================================
# -- Sensor Callback -----------------------------------------------------------
# ==============================================================================
def process_image(image):
    global image_surface, opencv_frame
    # Convert raw sensor data to a numpy array in BGRA format
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    
    # For OpenCV: Convert BGRA to BGR
    opencv_frame = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    
    # For pygame: Convert BGRA to RGB (swap channels) and create a Surface
    rgb_array = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
    image_surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))

# ==============================================================================
# -- Camera Transform Update ---------------------------------------------------
# ==============================================================================
def update_camera_transform(vehicle, camera, distance=7.0, height=3.0):
    """
    Update the camera's transform so that it follows the vehicle from behind and above.
    """
    veh_transform = vehicle.get_transform()
    forward_vector = veh_transform.get_forward_vector()
    # Position the camera behind the vehicle
    cam_location = veh_transform.location - forward_vector * distance
    cam_location.z += height

    # Compute the rotation so that the camera looks at the vehicle
    direction = veh_transform.location - cam_location
    yaw = math.degrees(math.atan2(direction.y, direction.x))
    pitch = math.degrees(math.atan2(direction.z, math.sqrt(direction.x**2 + direction.y**2)))
    cam_rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0)
    cam_transform = carla.Transform(cam_location, cam_rotation)
    camera.set_transform(cam_transform)

# ==============================================================================
# -- Main Loop -----------------------------------------------------------------
# ==============================================================================
def main():
    global image_surface, opencv_frame, prev_frame_time

    # Initialize pygame (gameplay & telemetry window)
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA: Gameplay, Telemetry & Detection")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    
    # Connect to CARLA simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Spawn a vehicle (e.g., Tesla Model 3) at a random spawn point
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Spawned vehicle at:", spawn_point.location)

    # Spawn an RGB camera sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    init_transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation())
    camera = world.spawn_actor(camera_bp, init_transform)
    camera.listen(process_image)
    print("Camera sensor spawned.")

    # Create an OpenCV window for detection display
    cv2.namedWindow("CARLA OpenCV Detection", cv2.WINDOW_AUTOSIZE)

    # Flag for autopilot mode
    auto_mode = False

    running = True
    while running:
        # Process pygame events (window events and key inputs)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                # Toggle autopilot mode with the A key
                elif event.key == K_a:
                    auto_mode = not auto_mode
                    vehicle.set_autopilot(auto_mode)
                    print("Autopilot mode set to:", auto_mode)

        # If autopilot is off, process manual control inputs
        if not auto_mode:
            keys = pygame.key.get_pressed()
            control = carla.VehicleControl()
            if keys[K_UP]:
                control.throttle = 0.5
            if keys[K_DOWN]:
                control.brake = 1.0
            if keys[K_LEFT]:
                control.steer = -0.5
            if keys[K_RIGHT]:
                control.steer = 0.5
            # Optionally enable reverse if Q is pressed
            elif keys[K_q]:
                control.reverse = True
            vehicle.apply_control(control)
        
        # Update camera transform so it follows the vehicle
        update_camera_transform(vehicle, camera)

        # Retrieve telemetry: speed and location
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        transform = vehicle.get_transform()
        location = transform.location

        # Update the pygame window: display gameplay (camera view) and overlay telemetry
        if image_surface is not None:
            display.blit(image_surface, (0, 0))
        else:
            display.fill((50, 50, 50))
        speed_text = font.render("Speed: {:.2f} m/s".format(speed), True, (255, 255, 255))
        location_text = font.render("Location: ({:.1f}, {:.1f}, {:.1f})".format(location.x, location.y, location.z), True, (255, 255, 255))
        autopilot_text = font.render("Autopilot: {}".format(auto_mode), True, (255, 255, 255))
        display.blit(speed_text, (10, 10))
        display.blit(location_text, (10, 30))
        display.blit(autopilot_text, (10, 50))
        pygame.display.flip()

        # --- Object Detection with OpenCV ---
        if opencv_frame is not None:
            # Make a copy of the current frame for detection processing
            detection_frame = opencv_frame.copy()
            
            # Compute FPS for detection display
            new_frame_time = time.time()
            fps = 1.0 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0.0
            prev_frame_time = new_frame_time
            fps_text = str(int(fps))
            font_cv = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(detection_frame, fps_text, (7, 70), font_cv, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
            # Run object detection
            classIndex, confidence, bbox = model.detect(detection_frame, confThreshold=0.65)
            if len(classIndex) != 0:
                for classInd, box in zip(classIndex.flatten(), bbox):
                    cv2.rectangle(detection_frame, box, (255, 0, 0), 2)
                    if 1 <= classInd <= 80:
                        cv2.putText(detection_frame, classLabels[classInd - 1], 
                                    (box[0] + 10, box[1] + 40), font_cv, 1, (0, 255, 0), 2)
            cv2.imshow("CARLA OpenCV Detection", detection_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

        clock.tick(60)  # Limit the loop to 60 FPS

    # Cleanup actors and close windows
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    pygame.quit()
    cv2.destroyAllWindows()
    print("Cleaned up and exiting.")

if __name__ == '__main__':
    main()
