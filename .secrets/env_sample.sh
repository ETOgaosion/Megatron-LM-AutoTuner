#!/bin/bash

BASE_DATA_DIR=${BASE_DATA_DIR:-"/data/common"}

export NCCL_SOCKET_IFNAME=ens37f0

HFD_LOCATION=${HFD_LOCATION:-"${BASE_DATA_DIR}/softwares/hfd.sh"}