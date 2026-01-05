@echo off
docker run --rm -it ^
  --platform linux/amd64 ^
  --name gfr ^
  -e DISPLAY=host.docker.internal:0.0 ^
  -e LIBGL_ALWAYS_INDIRECT=0 ^
  -e GALLIUM_DRIVER=llvmpipe ^
  -v "%cd%":/workspace_yuykim ^
  -w /workspace_yuykim ^
  hisplan/gfootball:2.10.2-facamp.1