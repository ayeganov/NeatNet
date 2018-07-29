DOCKER_NAME="neatnet_dev"
xhost +local:root 1> /dev/null 2>&1

docker exec \
    -u ${USER} \
    -it ${DOCKER_NAME} \
    /bin/bash

xhost -local:root 1> /dev/null 2>&1
