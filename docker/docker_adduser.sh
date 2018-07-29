#!/usr/bin/env bash

###############################################################################
#   +----------------------------------------------------------------------+
#   | Copyright (c) 2017-present iSee, Inc. (http://www.isee.ai)           |
#   +----------------------------------------------------------------------+
###############################################################################

addgroup --force-badname --gid "$DOCKER_GRP_ID" "$DOCKER_GRP"
adduser --disabled-password --gecos '' "$DOCKER_USER" \
    --uid "$DOCKER_USER_ID" --gid "$DOCKER_GRP_ID" 2>/dev/null
usermod -aG sudo "$DOCKER_USER"
echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
cp -r /etc/skel/. /home/${DOCKER_USER}

#echo 'if [ -e "/beyond/scripts/base.sh" ]; then source /beyond/scripts/base.sh; fi' >> "/home/${DOCKER_USER}/.bashrc"
echo 'export QT_X11_NO_MITSHM=1' >> "/home/${DOCKER_USER}/.bashrc"
chown -R ${DOCKER_USER}:${DOCKER_GRP} "/home/${DOCKER_USER}"
