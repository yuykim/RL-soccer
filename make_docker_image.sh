#!/bin/bash
set -e

fname="2.10.2-facamp.1"

wget -O v${fname}.zip https://github.com/hisplan/football/archive/refs/tags/v${fname}.zip

unzip v${fname}.zip
mv football-${fname} gfootball

docker build --platform linux/amd64 \
    -t gfootball .

docker tag gfootball:latest gfootball:2.10.2-facamp.1
