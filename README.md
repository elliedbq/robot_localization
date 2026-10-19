# Robot Localization

Making a particle filter that uses a given world map, robot odometry, and laser scans to find the robot's position in the world.

## Conceptual Overview
### Workflow
add picture? or wtv we write in report

### 1. Initialize the Particle Cloud
`initialize_particle_cloud()`

Determine a set of random particles for the particle cloud.

### 2. Predict: Motion Update
`update_particles_with_odom()`

After each robot movement, update each particle and add any
noise from the odometry measurement. This also makes sure the
filter does not collapse onto a single particle location.
 
### 3. Guess: Observation Update
`update_particles_with_laser`

Use the laser scan to determine the confidence value (weight) of each particle.

### 4. Estimate Pose
`update_robot_pose()`

Using the particle with the highest confidence value (weight), approximate where the robot is on the map. 

### 5. Resampling
`resample_particles()`

Using the weights (probability) of each particle being correct position of the robot, resample the particles, adding a random injection of particles in case of a large convergence, rooms, or symmetrical obstacles.

### More Details
Look at report: 

## How to Run

1. Download or clone our repository in your ROS workspace: `ros2_ws/src/`

2. Build and compile the packages:

        colcon build --symlink-install

3. Source the `install.bash` script:

        source ~/ros2_ws/install/setup.bash

4. Launch rviz2 and pass configuration file:

         rviz2 -d ~/ros2_ws/src/robot_localization/rviz/turtlebot_bag_files.rviz

5. Launch particle filter and map, changing `username` and making sure the file path to the map is correct:

        ros2 launch robot_localization test_pf.py map_yaml:=/home/username/ros2_ws/src/robot_localization/maps/mac_1st_floor_final.yaml

6. Play the bag file containing the robot's motion and laser scan data, changing `username` and making sure the file path to the bag file is correct:

        ros2 launch robot_localization test_pf.py map_yaml:=/home/username/ros2_ws/src/robot_localization/maps/mac_1st_floor_final.yaml

    Alternatively, run a teleop on a physical robot and move the robot around and gather laser scan data from there. 

