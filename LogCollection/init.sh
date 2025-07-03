#!/bin/bash

# exit if the script is not running as root
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

version_check(){
    MAJOR_VERSION=$(uname -r | awk -F '.' '{print $1}')
    MINOR_VERSION=$(uname -r | awk -F '.' '{print $2}')
    if [ $MAJOR_VERSION -ge 5 ] && [ $MINOR_VERSION -gt 15 ] || [ $MAJOR_VERSION -ge 6 ] ; then
        echo 1
    else
        echo 0
    fi
}

if [ `version_check` -eq 0 ]; then
    apt-get install linux-image-5.15.0-89-generic -y
fi

sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# install python
python3 --version > /dev/null 2>&1
if [ "$?" -ne 0 ]
then
    echo "Install python3"
    apt install python3 -y
fi

pip3 --version > /dev/null 2>&1
if [ "$?" -ne 0 ]
then
    echo "Install python3-pip"
    apt install python3-pip -y
fi

python3 -c 'import filelock' > /dev/null 2>&1
if [ $? -ne 0 ]; then
    pip3 install filelock
fi

# install stress-ng
stress-ng --version > /dev/null 2>&1
if [ "$?" -ne 0 ]
then
    echo "Install stress-ng"
    apt install stress-ng -y
fi

# install jq to modify json
jq --version > /dev/null 2>&1
if [ "$?" -ne 0 ]
then
    echo "Install jq"
    apt install jq -y
fi

yq --version > /dev/null 2>&1
if [ "$?" -ne 0 ]
then
    echo "Install yq"
    snap install yq
fi

apt install expect -y

criu check
if [ $? -ne 0 ]; then
    apt install pkg-config libnet-dev python-yaml libaio-dev libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler python-protobuf libnl-3-dev libcap-dev libnftables-dev build-essential -y
    wget http://github.com/checkpoint-restore/criu/archive/v3.19/criu-3.19.tar.gz
    tar xf criu-3.19.tar.gz
    cd criu-3.19
    make -j2
    cp criu/criu /usr/bin
    mkdir -p /etc/criu
    echo tcp-established >> /etc/criu/runc.conf
    echo file-locks >> /etc/criu/runc.conf
fi

# install docker
# you must restart terminal
docker --version > /dev/null 2>&1
if [ "$?" -ne 0 ]
then
    echo "Install docker"
    apt-get remove docker docker-engine docker.io containerd runc -y > /dev/null 2>&1
    apt-get update
    apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release -y

    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y

    echo "{ \"experimental\": true }" > /etc/docker/daemon.json
    systemctl restart docker

    for i in $(awk -F: '{ if ($3>= 1000 && $3!=65534  ) print $1 }' /etc/passwd)
    do
      usermod -aG docker $i
    done
fi
