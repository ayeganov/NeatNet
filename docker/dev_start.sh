#!/usr/bin/env bash

DOCKER_NAME="neatnet_dev"
DOCKER_WORK_DIR="/neatnet"
if [ -z "${DOCKER_REPO}" ]; then
    DOCKER_REPO=ayeganov/neatnet
fi

VERSION=$(docker images | grep ${DOCKER_REPO} | awk '{print $2}')
echo "Version: ${VERSION}"

IMG=${DOCKER_REPO}:${VERSION}
LOCAL_DIR="$( cd "$( dirname "$0" )/.." && pwd )"

echo "Local dir: '${LOCAL_DIR}'"
echo "Img: '${IMG}'"

USER_ID=$(id -u)
GRP=$(id -g -n)
GRP_ID=$(id -g)
LOCAL_HOST=`hostname`
DOCKER_HOME="/home/$USER"
if [ "$USER" == "root" ];then
    DOCKER_HOME="/root"
fi
if [ ! -d "$HOME/.cache" ];then
    mkdir "$HOME/.cache"
fi

function find_device() {
    # ${1} = device pattern
    local device_list=$(find /dev -name "${1}*")
    if [ -n "${device_list}" ]; then
        local devices=""
        for device in $(find /dev -name "${1}"); do
            echo "Found device: ${device}."
            devices="${devices} --device ${device}:${device}"
        done
        echo "${devices}"
    fi
}

devices="$(find_device video)"
echo "Devices: '${devices}'"

docker run -it \
    -d \
    --name ${DOCKER_NAME} \
    -e DISPLAY=${DISPLAY} \
    -e DOCKER_USER=$USER \
    -e USER=$USER \
    -e DOCKER_USER_ID=$USER_ID \
    -e DOCKER_GRP=$GRP \
    -e DOCKER_GRP_ID=$GRP_ID \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $LOCAL_DIR:${DOCKER_WORK_DIR} \
    -v /media:/media \
    --privileged -v /dev/bus/usb:/dev/bus/usb \
    -v $HOME/.cache:${DOCKER_HOME}/.cache \
    -v /etc/localtime:/etc/localtime:ro \
    --net host \
    -w ${DOCKER_WORK_DIR} \
    ${devices} \
    --add-host in_dev_docker:127.0.0.1 \
    --add-host ${LOCAL_HOST}:127.0.0.1 \
    --hostname in_dev_docker \
    --shm-size 4096M \
    $IMG
if [ "${USER}" != "root" ]; then
    docker exec ${DOCKER_NAME} bash -c "${DOCKER_WORK_DIR}/docker/docker_adduser.sh"
fi
