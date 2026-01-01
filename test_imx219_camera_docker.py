import cv2

gst_pipeline = (
    "nvarguscamerasrc sensor-id=0 sensor-mode=4 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1, format=NV12 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "queue ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=1 sync=false max-buffers=1"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

print("opened:", cap.isOpened())
ret, frame = cap.read()
print("read:", ret, None if not ret else frame.shape)

cap.release()