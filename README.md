# cs4391_spring26

# docker commands

xhost +local:

sudo docker run -it --rm \
  --ipc=host --runtime=nvidia \
  --network host \
  --device=/dev/video1:/dev/video1 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/yuxiang/Projects/cs4391_spring26:/ultralytics/cs4391_spring26 \
  -v /tmp/argus_socket:/tmp/argus_socket \
  ultralytics/ultralytics:latest-jetson-jetpack6-opencv


# Debug opencv
OPENCV_LOG_LEVEL=DEBUG

# YOLO

https://www.jetson-ai-lab.com/archive/tutorial_ultralytics.html

sudo docker run -it --ipc=host --runtime=nvidia -v /home/yuxiang/Projects/cs4391_spring26:/ultralytics/cs4391_spring26 ultralytics/ultralytics:latest-jetson-jetpack6 