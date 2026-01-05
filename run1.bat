@echo off
docker run -it --rm ^
  --platform linux/amd64 ^
  --name gfootball_x11 ^
  -e DISPLAY=host.docker.internal:0.0 ^
  -v "%cd%":/workspace_yuykim ^
  -w /workspace_yuykim ^
  hisplan/gfootball:2.10.2-facamp.1