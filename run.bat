@echo off

docker run --rm -it ^
    --platform linux/amd64 ^
    --name gfr ^
    -v "%cd%":/workspace_yuykim ^
    -w /workspace_yuykim ^
    hisplan/gfootball:2.10.2-facamp.1