from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
import pygame
import numpy as np
import random
import math
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ESCAPE, K_a, K_q

# Global variable to store the latest camera image as a pygame Surface
image_surface = None

def process_image(image):
    global image_surface
    # Convert the raw sensor data to a NumPy array with shape (height, width, 4)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    # Drop the alpha channel and convert from BGRA to RGB (swap channels)
    array = array[:, :, :3][:, :, ::-1]
    # Create a pygame Surface (transpose dimensions because pygame expects (width, height))
    image_surface = pygame.surfarray.make_surface(np.transpose(array, (1, 0, 2)))

def update_camera_transform(vehicle, camera, distance=7.0, height=3.0):
    """
    Update the camera's transform so that it follows the vehicle from behind and above.
    """
    veh_transform = vehicle.get_transform()
    forward_vector = veh_transform.get_forward_vector()
    # Position the camera behind the vehicle
    cam_location = veh_transform.location - forward_vector * distance
    cam_location.z += height

    # Compute the direction from the camera to the vehicle
    direction = veh_transform.location - cam_location
    yaw = math.degrees(math.atan2(direction.y, direction.x))
    pitch = math.degrees(math.atan2(direction.z, math.sqrt(direction.x**2 + direction.y**2)))
    cam_rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0)
    cam_transform = carla.Transform(cam_location, cam_rotation)
    camera.set_transform(cam_transform)

def main():
    global image_surface

    # Initialize pygame and create a window
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA Auto Driving Mode Toggle")
    clock = pygame.time.Clock()

    # Initialize a font for displaying vehicle info
    font = pygame.font.SysFont("Arial", 18)
    
    # Connect to the CARLA simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Choose a vehicle blueprint and spawn the vehicle at a random spawn point
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Spawned vehicle at:", spawn_point.location)

    # Spawn a standalone RGB camera sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    # Initial dummy transform; it will be updated immediately
    init_transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation())
    camera = world.spawn_actor(camera_bp, init_transform)
    camera.listen(lambda image: process_image(image))
    print("Camera sensor spawned.")

    # Flag for autopilot mode
    auto_mode = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                # Toggle autopilot mode when the A key is pressed
                elif event.key == K_a:
                    auto_mode = not auto_mode
                    vehicle.set_autopilot(auto_mode)
                    print("Autopilot mode set to:", auto_mode)
                    
        # If autopilot is off, use manual control with arrow keys
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
            elif keys[K_q]:
                control.reverse = True
            vehicle.apply_control(control)

        # Update the third-person camera to follow the vehicle
        update_camera_transform(vehicle, camera)

        # Retrieve vehicle speed and location data
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        transform = vehicle.get_transform()
        location = transform.location

        # Render the latest camera image in the pygame window
        display.fill((0, 0, 0))
        if image_surface is not None:
            display.blit(image_surface, (0, 0))

        # Create text surfaces for speed and location
        speed_text = font.render("Speed: {:.2f} m/s".format(speed), True, (255, 255, 255))
        location_text = font.render("Location: ({:.1f}, {:.1f}, {:.1f})".format(location.x, location.y, location.z), True, (255, 255, 255))
        autopilot_text = font.render("Autopilot: {}".format(auto_mode), True, (255, 255, 255))
        
        # Overlay the text onto the display
        display.blit(speed_text, (10, 10))
        display.blit(location_text, (10, 30))
        display.blit(autopilot_text, (10, 50))
        
        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS


    # Cleanup actors and quit
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    pygame.quit()
    print("Cleaned up and exiting.")

if __name__ == '__main__':
    main()
