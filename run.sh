#!/bin/bash

docker run --rm -it \
    --platform linux/amd64 \
    --name gfr \
    -v "/$(pwd):/workspace" \
    hisplan/gfootball:2.10.2-facamp.1