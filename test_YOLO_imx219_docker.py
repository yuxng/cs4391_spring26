import cv2
from ultralytics import YOLO

## sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module if seeing
## Gtk-Message: 10:34:56.086: Failed to load module "canberra-gtk-module"

# Load the exported TensorRT model
trt_model = YOLO("yolov8n.engine")

gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Cannot open IMX219 camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = trt_model(frame)        

    # Annotated visualization frame (BGR)
    annotated = results[0].plot()

    cv2.imshow("IMX219-YOLO", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()