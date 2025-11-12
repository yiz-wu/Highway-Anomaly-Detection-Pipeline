# Start from the fully built Lanelet2 image you already have
FROM lanelet2:latest

# Switch to developer user
USER developer
WORKDIR /home/developer/app

COPY --chown=developer:developer ./ ./

RUN sudo apt-get update && sudo apt-get install -y python3-venv libgl1 libglib2.0-0 && \
    python3 -m venv linux_venv && \
    source linux_venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements_cv.txt && \
    echo 'export PYTHONPATH=$PYTHONPATH:/home/developer/workspace/devel/lib/python3/dist-packages' >> linux_venv/bin/activate

# Auto-source ROS and workspace in any shell
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source ~/workspace/devel/setup.bash" >> ~/.bashrc

# Default workdir
WORKDIR /home/developer/app

# Default command: activate venv and execute python
ENTRYPOINT ["/bin/bash", "-c", "source linux_venv/bin/activate && python \"$@\"", "--"]
