# Set variables
IMAGE_NAME="dipy" # Change after confirmation from Serge
TAG="latest"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully: ${IMAGE_NAME}:${TAG}"
else
    echo "Error: Docker image build failed"
    exit 1
fi