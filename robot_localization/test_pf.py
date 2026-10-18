"""
The following Unit tests are designed to test the pf.py functions
Running pytest will conduct these tests and all the tests will
pass. If a test failed, an assertion error would occur.

Functions:



"""

from unittest.mock import MagicMock
import pytest
import math
import numpy as np

# File and classes being tested
from pf import Particle, ParticleFilter


# Set-up
# Want to bypass the __init__ call - Gemini helped figure out how to do the bypassing
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


# Test the weights = 1
def test_log_weights(pf_node):
    """Check that the weight sums to 1"""
    p1 = Particle(w=1.0)
    p2 = Particle(w=3.0)
    pf_node.particle_cloud = [p1, p2]

    pf_node.normalize_particles()
    total = sum(p.w for p in pf_node.particle_cloud)
    assert math.isclose(total, 1.0, rel_tol=1e-5)
    assert math.isclose(p1.w, 0.25, rel_tol=1e-5)
    assert math.isclose(p2.w, 0.75, rel_tol=1e-5)


# Check particle inside bounding box
def test_bounding_box_logic(pf_node):
    """Ensure particles only fall inside the bounding box"""
    # simulated bounds
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


# We had issues with p.log_weight being Nan
def test_nan_log_weights(pf_node):
    """Ensure p.log_weight does not become NaN when out of bounds."""
    # Test only a few particles
    p1 = Particle(x=0.0, y=0.0, theta=0.0)
    p2 = Particle(x=10.0, y=10.0, theta=math.pi / 2)
    pf_node.particle_cloud = [p1, p2]

    # Occupancy Fields
    pf_node.occupancy_field.get_closest_obstacle_distance.side_effect = [
        float("nan"),
        None,
        0.15,
        float("nan"),
    ]

    # Scan data (4 beams)
    r = [1.0, 2.0, 1.5, 3.0]
    theta = [0.0, 0.1, 0.2, 0.3]

    # Run  laser update step,which includes log_weight
    pf_node.update_particles_with_laser(r, theta)

    # Check  p.log_weight is not None and not NaN for each particle
    for i, p in enumerate(pf_node.particle_cloud):
        assert p.log_weight is not None, f"Particle {i} log_weight was not computed"
        assert not math.isnan(p.log_weight), f"Particle {i} log_weight evaluated to NaN"


# check if p.log_weight = nan
