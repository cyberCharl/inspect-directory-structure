FROM ubuntu:24.04

# Install apt packages
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y python3 python3-pip python3-dev python3-venv curl lsof

# set up virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python packages
# Any python package versions that are required must be pinned
RUN pip install requests==2.31.0

# Mock sudo
RUN echo -e '#!/bin/sh\nexec "$@"' > /usr/bin/sudo && chmod +x /usr/bin/sudo

# Specify the command to run when the container starts
CMD ["/bin/bash"]
