#!/usr/bin/env bash

# Startup script for AWS

# Make the /mnt directory writable
sudo chown ubuntu:ubuntu /mnt

# pip
pip install awscli

pip install -r requirements.txt
