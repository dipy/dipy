# Use an official Python runtime as a parent image
# We are not using slim due to hdf5 dependency
FROM python:3.10

# Install system dependencies and clean up in one RUN command
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install python dependencies
# We are installing numpy < 2 because some dependencies are complied with 1.x.x and they aren't working with 2.x.x
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    dipy \
    nibabel \
    scipy \
    matplotlib