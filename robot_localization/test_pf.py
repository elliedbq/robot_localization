"""
This file includes unit tests to test the pf.py functions
Running pytest in terminal will show if all tests passed,
and if one fails an assert message should appear.
"""

from unittest.mock import MagicMock
import pytest
import math
import numpy as np

# File and classes being tested
from pf import Particle, ParticleFilter


# Set-up
# Want to bypass the __init__ call - Gemini helped figure out how to do the bypassing and explained MagicMock()
@pytest.fixture
def pf_node():
    """
    Create an instance of ParticleFilter to allow testing of logic without needing live nodes.
    """

    node = ParticleFilter.__new__(ParticleFilter)

    # Initialize basic algorithmic parameters
    node.d_thresh = 0.2
    node.a_thresh = math.pi / 6
    node.sigma_hit = 0.2
    node.n_particles = 4
    node.current_odom_xy_theta = [0.0, 0.0, 0.0]

    # Mock ROS/Helper dependencies
    node.occupancy_field = MagicMock()
    node.transform_helper = MagicMock()
    node.particle_cloud = []

    return node


# Test that the particles make sense without log_weight set. Should equal 1
def test_particles_standard_weights(pf_node):
    """Test that the weights equal 1."""
    pf_node.n_particles = 3
    pf_node.particle_cloud = [Particle(w=2.0), Particle(w=3.0), Particle(w=5.0)]
    # Set log_weight to none
    for p in pf_node.particle_cloud:
        p.log_weight = None

    pf_node.normalize_particles()

    # The expected should sum to 1.0
    assert math.isclose(pf_node.particle_cloud[0].w, 0.2, abs_tol=1e-5)
    assert math.isclose(pf_node.particle_cloud[1].w, 0.3, abs_tol=1e-5)
    assert math.isclose(pf_node.particle_cloud[2].w, 0.5, abs_tol=1e-5)
    assert math.isclose(sum(p.w for p in pf_node.particle_cloud), 1.0, abs_tol=1e-5)


# Now test with log_weight
def test_log_weights(pf_node):
    """Check log_weight normalization functions correctly and sums to 1."""
    p1 = Particle()
    p2 = Particle()

    # Log(3) ~ 1.0986
    # P2 should be about 3x p1
    p1.log_weight = -10.0
    p2.log_weight = -10.0 + math.log(3.0)
    pf_node.particle_cloud = [p1, p2]

    pf_node.normalize_particles()

    total = sum(p.w for p in pf_node.particle_cloud)
    assert math.isclose(total, 1.0, rel_tol=1e-5)
    assert math.isclose(p1.w, 0.25, rel_tol=1e-5)  # 1 / (1 + 3)
    assert math.isclose(p2.w, 0.75, rel_tol=1e-5)  # 3 / (1 + 3)


# Check particle inside bounding box and that the amount of particles is actually consistent
def test_bounding_box_logic(pf_node):
    """Ensure particles only fall inside the bounding box"""
    # Need simulated bounds
    bounds = ((-5.0, 5.0), (-5.0, 5.0))
    pf_node.occupancy_field.get_obstacle_bounding_box.return_value = bounds
    (min_x, max_x), (min_y, max_y) = bounds

    # Use MagicMock to keep things simulated
    pf_node.odom_pose = MagicMock()
    pf_node.transform_helper.convert_pose_to_xy_and_theta.return_value = [0.0, 0.0, 0.0]
    pf_node.publish_particles = MagicMock()

    pf_node.n_particles = 100  # Amount of particles
    mock_timestamp = MagicMock()
    pf_node.initialize_particle_cloud(mock_timestamp)

    # Checks particle_cloud still same amount
    assert len(pf_node.particle_cloud) == 100

    # Check that particles are in the bounds
    for p in pf_node.particle_cloud:
        assert (
            min_x <= p.x <= max_x
        ), f"Particle x={p.x} out of bounds [{min_x}, {max_x}]"
        assert (
            min_y <= p.y <= max_y
        ), f"Particle y={p.y} out of bounds [{min_y}, {max_y}]"


# Want to check the heading of the particles updates as expected
def test_particle_translation(pf_node):
    """
    Test that particle translation and orientation makes sense.
    A forward movement in the robot's local frame should then be:
    - Theta = 0 particle moves along the global X-axis.
    - Theta = pi/2 particle moves along the global Y-axis."""
    pf_node.occupancy_field.get_obstacle_bounding_box.return_value = (
        (-10, 10),
        (-10, 10),
    )

    # two particles with different thetas
    p_east = Particle(x=0.0, y=0.0, theta=0.0, w=0.5)
    p_north = Particle(x=0.0, y=0.0, theta=math.pi / 2, w=0.5)
    pf_node.particle_cloud = [p_east, p_north]
    pf_node.current_odom_xy_theta = [0.0, 0.0, 0.0]

    # Have to mimic the robot movement
    pf_node.odom_pose = MagicMock()
    pf_node.transform_helper.convert_pose_to_xy_and_theta.return_value = [1.0, 0.0, 0.0]

    # Stop random noise to allow checking of just the particle translation
    orig_normal = np.random.normal
    np.random.normal = lambda loc=0.0, scale=1.0, size=None: 0.0
    try:
        pf_node.update_particles_with_odom()
    finally:
        np.random.normal = orig_normal  # Back to original function

    assert math.isclose(p_east.x, 1.0, abs_tol=1e-3)
    assert math.isclose(p_east.y, 0.0, abs_tol=1e-3)

    assert math.isclose(p_north.x, 0.0, abs_tol=1e-3)
    assert math.isclose(p_north.y, 1.0, abs_tol=1e-3)


# We had issues with p.log_weight being Nan
def test_nan_log_weights(pf_node):
    """Check that p.log_weight is not Nan."""
    # Only testing a few particles
    p1 = Particle(x=0.0, y=0.0, theta=0.0)
    p2 = Particle(x=10.0, y=10.0, theta=math.pi / 2)
    pf_node.particle_cloud = [p1, p2]

    # Occupancy field
    pf_node.occupancy_field.get_closest_obstacle_distance.side_effect = [
        float("nan"),
        None,
        0.15,
        float("nan"),
    ]

    # Scan the data this is only with 4 beams
    r = [1.0, 2.0, 1.5, 3.0]
    theta = [0.0, 0.1, 0.2, 0.3]

    # Run update_particles_with_laser which includes p.log_weight
    pf_node.update_particles_with_laser(r, theta)

    # Check p.log_weight is not None and not NaN for each particle
    for i, p in enumerate(pf_node.particle_cloud):
        assert p.log_weight is not None, f"Particle {i} log_weight was not computed"
        assert not math.isnan(p.log_weight), f"Particle {i} log_weight evaluated to NaN"
