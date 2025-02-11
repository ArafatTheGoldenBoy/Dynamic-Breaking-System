from __future__ import print_function

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import glob
import os
import sys

try:
    sys.path.append(glob.glob(
        '../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name=='nt' else 'linux-x86_64')
    )[0])
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
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ESCAPE, K_a, K_q

# Global variables for the camera image in both formats
image_surface = None   # pygame Surface for gameplay window
opencv_frame = None    # OpenCV BGR image for OpenCV window

def process_image(image):
    global image_surface, opencv_frame
    # Convert raw sensor data to a numpy array (BGRA format)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    
    # For OpenCV: Convert BGRA to BGR and store in opencv_frame
    opencv_frame = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    
    # For pygame: Convert to RGB (swap channels) and create a Surface
    rgb_array = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
    image_surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))

def update_camera_transform(vehicle, camera, distance=7.0, height=3.0):
    """
    Update the camera's transform so that it follows the vehicle from behind and above.
    """
    veh_transform = vehicle.get_transform()
    forward_vector = veh_transform.get_forward_vector()
    cam_location = veh_transform.location - forward_vector * distance
    cam_location.z += height

    # Compute rotation so the camera looks at the vehicle
    direction = veh_transform.location - cam_location
    yaw = math.degrees(math.atan2(direction.y, direction.x))
    pitch = math.degrees(math.atan2(direction.z, math.sqrt(direction.x**2 + direction.y**2)))
    cam_rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0)
    cam_transform = carla.Transform(cam_location, cam_rotation)
    camera.set_transform(cam_transform)

def main():
    global image_surface, opencv_frame

    # Initialize pygame window (for gameplay & telemetry)
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA: Gameplay & Telemetry")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    
    # Connect to the CARLA simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    # Spawn the vehicle
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Spawned vehicle at:", spawn_point.location)
    
    # Spawn the camera sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    init_transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation())
    camera = world.spawn_actor(camera_bp, init_transform)
    camera.listen(process_image)
    print("Camera sensor spawned.")
    
    # Create an OpenCV window (all OpenCV GUI functions must be in the main thread)
    cv2.namedWindow("CARLA OpenCV", cv2.WINDOW_AUTOSIZE)
    
    # Flag for autopilot mode
    auto_mode = False

    running = True
    while running:
        # Process pygame events for control and window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                # Toggle autopilot with the A key
                elif event.key == K_a:
                    auto_mode = not auto_mode
                    vehicle.set_autopilot(auto_mode)
                    print("Autopilot mode set to:", auto_mode)
        
        # Manual control if autopilot is off
        if not auto_mode:
            keys = pygame.key.get_pressed()
            control = carla.VehicleControl()
            if keys[K_UP]:
                control.throttle = 1.0
            if keys[K_DOWN]:
                control.brake = 1.0
            if keys[K_LEFT]:
                control.steer = -0.5
            if keys[K_RIGHT]:
                control.steer = 0.5
            # Optionally, enable reverse if Q is pressed
            elif keys[K_q]:
                control.reverse = True
            vehicle.apply_control(control)
        
        # Update camera transform so it follows the vehicle
        update_camera_transform(vehicle, camera)
        
        # Retrieve telemetry data
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        transform = vehicle.get_transform()
        location = transform.location
        
        # Draw the camera image (gameplay) onto the pygame window
        if image_surface is not None:
            display.blit(image_surface, (0, 0))
        else:
            display.fill((50, 50, 50))
        
        # Overlay telemetry information on top
        speed_text = font.render("Speed: {:.2f} m/s".format(speed), True, (255, 255, 255))
        location_text = font.render("Location: ({:.1f}, {:.1f}, {:.1f})".format(location.x, location.y, location.z), True, (255, 255, 255))
        autopilot_text = font.render("Autopilot: {}".format(auto_mode), True, (255, 255, 255))
        display.blit(speed_text, (10, 10))
        display.blit(location_text, (10, 30))
        display.blit(autopilot_text, (10, 50))
        pygame.display.flip()
        
        # Also update the OpenCV window with the latest frame
        if opencv_frame is not None:
            cv2.imshow("CARLA OpenCV", opencv_frame)
        # Process OpenCV events; pressing 'q' here will also exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
        
        clock.tick(60)  # Limit to 60 FPS

    # Cleanup actors and close windows
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    pygame.quit()
    cv2.destroyAllWindows()
    print("Cleaned up and exiting.")

if __name__ == '__main__':
    main()
