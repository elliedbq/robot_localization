#!/usr/bin/env python3

"""This is the starter code for the robot localization project"""

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud, Particle
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from rclpy.duration import Duration
import math
import time
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler

from helper_functions import draw_random_sample


class Particle(object):
    """Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
    Attributes:
        x: the x-coordinate of the hypothesis relative to the map frame
        y: the y-coordinate of the hypothesis relative ot the map frame
        theta: the yaw of the hypothesis relative to the map frame
        w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """Construct a new Particle
        x: the x-coordinate of the hypothesis relative to the map frame
        y: the y-coordinate of the hypothesis relative ot the map frame
        theta: the yaw of KeyboardInterruptthe hypothesis relative to the map frame
        w: the particle weight (the class does not ensure that particle weights are normalized
        """
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

        self.raw_weight = None
        self.log_weight = None

    def as_pose(self):
        """A helper function to convert a particle to a geometry_msgs/Pose message"""
        q = quaternion_from_euler(0, 0, self.theta)
        return Pose(
            position=Point(x=self.x, y=self.y, z=0.0),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
        )

    # TODO: define additional helper functions if needed

    def check_pose(self, min_x, max_x, min_y, max_y):
        """checks if particle is within bounding box
        if not, re positions particle somewhere random"""

        if not (min_x < self.x < max_x and min_y < self.y < max_y):
            self.x = np.random.uniform(min_x, max_x)
            self.y = np.random.uniform(min_y, max_y)
            self.theta = np.random.uniform(-math.pi, math.pi)


class ParticleFilter(Node):
    """The class that represents a Particle Filter ROS Node
    Attributes list:
        base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
        map_frame: the name of the map coordinate frame (should be "map" in most cases)
        odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
        scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
        n_particles: the number of particles in the filter
        d_thresh: the amount of linear movement before triggering a filter update
        a_thresh: the amount of angular movement before triggering a filter update
        pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
        particle_pub: a publisher for the particle cloud
        last_scan_timestamp: this is used to keep track of the clock when using bags
        scan_to_process: the scan that our run_loop should process next
        occupancy_field: this helper class allows you to query the map for distance to closest obstacle
        transform_helper: this helps with various transform operations (abstracting away the tf2 module)
        particle_cloud: a list of particles representing a probability distribution over robot poses
        current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                               The pose is expressed as a list [x,y,theta] (where theta is the yaw)
        thread: this thread runs your main loop
    """

    def __init__(self):
        super().__init__("pf")
        self.base_frame = "base_footprint"  # the frame of the robot base
        self.map_frame = "map"  # the name of the map coordinate frame
        self.odom_frame = "odom"  # the name of the odometry coordinate frame
        self.scan_topic = "scan"  # the topic where we will get laser scans from

        self.n_particles = 1000  # the number of particles to use

        self.d_thresh = 0.2  # the amount of linear movement before performing an update
        self.a_thresh = (
            math.pi / 6
        )  # the amount of angular movement before performing an update

        # TODO: define additional constants if needed

        self.sigma_hit = 0.3  # should check this number. standard deviation for obstacle delta to particle weight

        print("new pf.py running")
        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.create_subscription(
            PoseWithCovarianceStamped, "initialpose", self.update_initial_pose, 10
        )

        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(
            ParticleCloud, "particle_cloud", qos_profile_sensor_data
        )

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # this is used to keep track of the timestamps coming from bag files
        # knowing this information helps us set the timestamp of our map -> odom
        # transform correctly
        self.last_scan_timestamp = None
        # this is the current scan that our run_loop should process
        self.scan_to_process = None
        # your particle cloud will go here
        self.particle_cloud = []

        self.current_odom_xy_theta = []
        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)

        # we are using a thread to work around single threaded execution bottleneck
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)

    def pub_latest_transform(self):
        """This function takes care of sending out the map to odom transform"""
        if self.last_scan_timestamp is None:
            return
        postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(
            seconds=0.1
        )
        self.transform_helper.send_last_map_to_odom_transform(
            self.map_frame, self.odom_frame, postdated_timestamp
        )

    def loop_wrapper(self):
        """This function takes care of calling the run_loop function repeatedly.
        We are using a separate thread to run the loop_wrapper to work around
        issues with single threaded executors in ROS2"""
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """This is the main run_loop of our particle filter.  It checks to see if
        any scans are ready and to be processed and will call several helper
        functions to complete the processing.

        You do not need to modify this function, but it is helpful to understand it.
        """
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process

        new_pose, delta_t = self.transform_helper.get_matching_odom_pose(
            self.odom_frame, self.base_frame, msg.header.stamp
        )
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # we will never get this transform, since it is before our oldest one
                self.scan_to_process = None
            return

        r, theta = self.transform_helper.convert_scan_to_polar_in_robot_frame(
            msg, self.base_frame
        )
        print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))
        # clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(
            self.odom_pose
        )
        print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
        elif not self.particle_cloud:
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud(msg.header.stamp)
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            # we have moved far enough to do an update!
            self.update_particles_with_odom()  # update based on odometry
            self.update_particles_with_laser(r, theta)  # update based on laser scan
            self.update_robot_pose()  # update robot's pose based on particles
            self.resample_particles()  # resample particles to focus on areas of high density

        self.pub_latest_transform()

    def moved_far_enough_to_update(self, new_odom_xy_theta):
        return (
            math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0])
            > self.d_thresh
            or math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1])
            > self.d_thresh
            or math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2])
            > self.a_thresh
        )

    def update_robot_pose(self):
        """Update the estimate of the robot's pose given the updated particles.
        There are two logical methods for this:
            (1): compute the mean pose
            (2): compute the most likely pose (i.e. the mode of the distribution)
        """
        # first make sure that the particle weights are normalized
        self.normalize_particles()

        # TODO: assign the latest pose into self.robot_pose as a geometry_msgs.Pose object
        # just to get started we will fix the robot's pose to always be at the origin
        self.robot_pose = Pose()

        # weight = 0
        # # find max weight and set its x and y coord to the predicted robot pose
        # for p in self.particle_cloud:
        #     if p.w > weight:
        #         weight = p.w
        #         largest_p = p

        # find weight of each particle but take into account the weight of surrounding particles as well
        surrounding_radius = 0.2  # random number rn
        max_weight = 0
        largest_p = self.particle_cloud[0]  # as fallback
        for i, p in enumerate(self.particle_cloud):
            weight = p.w
            # radius cluster thing
            # x_min = p.x - surrounding_radius
            # x_max = p.x + surrounding_radius
            # y_min = p.y - surrounding_radius
            # y_max = p.y + surrounding_radius
            # for j, p2 in enumerate(self.particle_cloud):
            #     if i != j:
            #         if x_min < p2.x < x_max and y_min < p2.y < y_max:
            #             weight += p2.w
            if weight > max_weight:
                largest_p = p
                max_weight = weight

        print("set robot pose")
        q = quaternion_from_euler(0, 0, largest_p.theta)
        self.robot_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.robot_pose.position = Point(x=largest_p.x, y=largest_p.y, z=0)

        if hasattr(self, "odom_pose"):
            self.transform_helper.fix_map_to_odom_transform(
                self.robot_pose, self.odom_pose
            )
        else:
            self.get_logger().warn(
                "Can't set map->odom transform since no odom data received"
            )

    def update_particles_with_odom(self):
        """Update the particles using the newly given odometry pose.
        The function computes the value delta which is a tuple (x,y,theta)
        that indicates the change in position and angle between the odometry
        when the particles were last updated and the current odometry.
        """
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(
            self.odom_pose
        )
        # compute the change in x,y,theta since our last update
        if (
            self.current_odom_xy_theta
        ):  # self.current_odom has to exist in order to find the difference
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (
                new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                new_odom_xy_theta[2] - self.current_odom_xy_theta[2],
            )

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        # TODO: modify particles using delta
        # particles have p.x and p.y and p.theta in world frame
        # robot moves in odom frame and delta is the change in odom frame

        # find relative change in location in world frame. (find delta in world frame)
        theta1r = old_odom_xy_theta[2]
        dxr = delta[0] * math.cos(theta1r) + delta[1] * math.sin(theta1r)
        dyr = -delta[0] * math.sin(theta1r) + delta[1] * math.cos(theta1r)

        # find wrapped angle difference( so theta does not get giant)
        dtheta = math.atan2(
            math.sin(new_odom_xy_theta[2] - old_odom_xy_theta[2]),
            math.cos(new_odom_xy_theta[2] - old_odom_xy_theta[2]),
        )

        # standard deviation for noise. adds abs value bc larger movements should have more noise.
        sigma_x = 0.01 + 0.1 * abs(dxr)
        sigma_y = 0.01 + 0.1 * abs(dyr)
        sigma_theta = 0.01 + 0.1 * abs(dtheta)

        # bounding box
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]

        # update particle with delta in world frame
        # - need to add rotation matrix in order to account for original heading of particle
        for p in self.particle_cloud:
            # add noise to local movement step in odom frame
            noise_x = float(np.random.normal(0, sigma_x))
            noise_y = float(np.random.normal(0, sigma_y))
            noise_theta = float(np.random.normal(0, sigma_theta))

            noisy_dxr = dxr + noise_x
            noisy_dyr = dyr + noise_y
            noisy_dtheta = dtheta + noise_theta

            p.x = p.x + math.cos(p.theta) * noisy_dxr - math.sin(p.theta) * noisy_dyr
            p.y = p.y + math.sin(p.theta) * noisy_dxr + math.cos(p.theta) * noisy_dyr
            # p.theta = p.theta + noisy_dtheta
            # will theta get too big? idk
            p.theta = math.atan2(
                math.sin(p.theta + noisy_dtheta), math.cos(p.theta + noisy_dtheta)
            )

            # check that particle is still in map. if not, give it a new random position.
            if not (min_x < p.x < max_x and min_y < p.y < max_y):
                print("not in map...")
                p.x = np.random.uniform(min_x, max_x)
                p.y = np.random.uniform(min_y, max_y)
                p.theta = np.random.uniform(-math.pi, math.pi)

        print(f"total particles: {len(self.particle_cloud)}")

    def resample_particles(self):
        """Resample the particles according to the new particle weights.
        The weights stored with each particle should define the probability that a particular
        particle is selected in the resampling step.  You may want to make use of the given helper
        function draw_random_sample in helper_functions.py.
        """
        # make sure the distribution is normalized
        self.normalize_particles()
        # TODO: fill out the rest of the implementation
        # draw_random_sample sets a lot of the actual logic up
        # need to have all probabilities from weights together - from p.w
        probabilities = [p.w for p in self.particle_cloud]

        # random_injection = 0
        random_injection = int(
            self.n_particles / 8
        )  # make ten percent of the new particles completely random
        # potentially add completely random particles ^

        sample_size = int(self.n_particles - random_injection)

        # resampling using draw_random_sample.
        # noise for new particles is added in the odometry displacement. (update_particles_with_odom)
        new_cloud = draw_random_sample(self.particle_cloud, probabilities, sample_size)

        for _ in range(0, random_injection):
            p = self.create_random_particle()
            new_cloud.append(p)

        self.particle_cloud = new_cloud
        # wipe weights
        for p in self.particle_cloud:
            p.w = 1 / self.n_particles
            p.log_weight = None

        self.publish_particles(self.last_scan_timestamp)

    def update_particles_with_laser(self, r, theta):
        """Updates the particle weights in response to the scan data
        r: the distance readings to obstacles
        theta: the angle relative to the robot frame for each corresponding reading
        """
        # TODO: implement this

        # throw out non finite data points from laser scan
        good_r = []
        good_theta = []
        for rad, the in zip(r, theta):
            if math.isfinite(rad) and rad > 0:
                good_r.append(rad)
                good_theta.append(the)

        for p in self.particle_cloud:
            step = 3  # checks scan at every step
            p.log_weight = 0
            for i in range(0, len(good_r), step):
                dist_xy = good_r[i]
                angle = good_theta[i]
                # find the obstacle location in world frame if robot was on particle
                beam_angle = p.theta + angle
                hit_x = p.x + dist_xy * math.cos(beam_angle)
                hit_y = p.y + dist_xy * math.sin(beam_angle)

                # compare obstacle location to occupancy field to see if
                # robots expectation of an obstacle matches to actual obstacle
                delta_obs = self.occupancy_field.get_closest_obstacle_distance(
                    hit_x, hit_y
                )
                if math.isfinite(delta_obs):
                    delta_obs = min(
                        delta_obs, 1.0
                    )  # cap on laser scan (any delta too large does not get kept)
                    error = -0.5 * ((delta_obs / self.sigma_hit) ** 2)
                else:
                    error = -2.0

                # log weight. no need to take into account old weight bc it needs to be reset each sample anyway
                p.log_weight += error
            print(p.log_weight)  # why is the log weight nan...

        self.normalize_particles()

    def update_initial_pose(self, msg):
        """Callback function to handle re-initializing the particle filter based on a pose estimate.
        These pose estimates could be generated by another ROS Node or could come from the rviz GUI
        """
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(msg.header.stamp, xy_theta)

    def initialize_particle_cloud(self, timestamp, xy_theta=None):
        """Initialize the particle cloud. only done in run_loop if no particle cloud exists.
        Arguments
        xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                  particle cloud around.  If this input is omitted, the odometry will be used
        """
        # odom is always (0,0,0)??
        # if xy_theta is None:
        #     xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(
        #         self.odom_pose
        #     )
        self.particle_cloud = []
        # TODO create particles

        # find map size
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]

        # initial weight is same
        initial_weight = 1.0 / self.n_particles

        # set standard deviation for making particles around xy_theta
        sigma_x = 0.2
        sigma_y = 0.2
        sigma_theta = 0.2

        if xy_theta is None:
            # equally distribute
            print("uniform distribution")
            for _ in range(0, self.n_particles):
                p = self.create_random_particle()  # create particle
                self.particle_cloud.append(p)
        else:
            # normal distribution over xy_theta
            print("yay normal distribution")
            for _ in range(0, self.n_particles):
                x = np.random.normal(xy_theta[0], sigma_x)
                y = np.random.normal(xy_theta[1], sigma_y)
                theta = np.random.normal(xy_theta[2], sigma_theta)

                p = Particle(x=x, y=y, theta=theta, w=initial_weight)  # create particle
                p.check_pose(min_x, max_x, min_y, max_y)
                self.particle_cloud.append(p)

        # self.normalize_particles() - not needed? in initialization, all the weights are already equal
        self.update_robot_pose()
        self.publish_particles(timestamp)
        print("particle cloud initialized")

    def create_random_particle(self, weight=None):
        # find map size
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]

        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        theta = np.random.uniform(-math.pi, math.pi)

        if weight is None:
            weight = 1 / self.n_particles

        p = Particle(x=x, y=y, theta=theta, w=weight)  # create particle
        return p

    def normalize_particles(self):
        """Make sure the particle weights define a valid distribution (i.e. sum to 1.0)"""
        # TODO: implement this

        total_weight = sum(p.w for p in self.particle_cloud)

        # logic within this if statement may be off - are we dividing by the right thing?

        if self.particle_cloud[0].log_weight is not None:
            max_log_weight = max(p.log_weight for p in self.particle_cloud)
            for p in self.particle_cloud:
                p.raw_weight = math.exp(p.log_weight - max_log_weight)

            total_raw_weight = sum(p.raw_weight for p in self.particle_cloud)
            for p in self.particle_cloud:
                p.w = p.raw_weight / total_raw_weight
        else:
            # if not, divide each weight by total_weight
            for p in self.particle_cloud:
                p.w = p.w / total_weight

        print("particles normalized")

    def publish_particles(self, timestamp):
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for p in self.particle_cloud:
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        self.last_scan_timestamp = msg.header.stamp
        # we throw away scans until we are done processing the previous scan
        # self.scan_to_process is set to None in the run_loop
        if self.scan_to_process is None:
            self.scan_to_process = msg


def main(args=None):
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
