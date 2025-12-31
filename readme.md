# Football Analytics Camp - January 2026
origin github : https://github.com/hisplan/facamp-2026-jan

나는 강화학습으로 축구한다

## Setup

### Prerequisites

You need to have Docker installed on your machine. You can download it from [here](https://www.docker.com/get-started).

### Option 1: Build Docker Image

```bash
./make_docker_image.sh
```

### Option 2: Download from Docker Hub

```bash
docker pull hisplan/gfootball:2.10.2-facamp.1
```

## How to Run

The following command will start a Docker container with the necessary environment:

```bash
./run.sh
```
or 
```bash
run.bat
```


Once inside the container, you can run the following command to check if everything is set up correctly:

```bash
python3 check.py
```

This should return the following message without errors:

```
gfootball env reset OK
```


