# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r hw5/requirements.txt
RUN cd hw5 && pip install -e .
RUN mkdir ws
# Make port 80 available to the world outside this container
EXPOSE 80

# Run tail -f /dev/null when the container launches
#CMD tail -f /dev/null