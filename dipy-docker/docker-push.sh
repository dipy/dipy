# Set variables
IMAGE_NAME="dipy" # Change after confirmation from Serge
TAG="latest"
REGISTRY="registry-url"  # Change after confirmation from Serge

# Tag the image for the registry
docker tag ${IMAGE_NAME}:${TAG} ${REGISTRY}/${IMAGE_NAME}:${TAG}

# Push the image to the registry
docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}

# Check if the push was successful
if [ $? -eq 0 ]; then
    echo "Docker image pushed successfully: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
else
    echo "Error: Failed to push Docker image"
    exit 1
fi