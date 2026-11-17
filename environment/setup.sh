#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Put ALL pip installs here, NOT in container.def.
# Only use container.def for the base Docker image and apt-get packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# Use: uv pip install --system <packages>
# Do NOT create a venv (python -m venv, virtualenv, etc.).
#
# mujoco: robot kinematics, Jacobians and signed distances (robot model and evaluator);
# clarabel: SOCP solver of the reference method.

set -e

uv pip install --system numpy==2.1.3 scipy==1.17.1 mujoco==3.3.0 clarabel==0.11.1
