import cv2
import time
from ultralytics import YOLO

## sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module if seeing
## Gtk-Message: 10:34:56.086: Failed to load module "canberra-gtk-module"

# Load the exported TensorRT model
trt_model = YOLO("yolov8n.engine")

# Try 0, then 1, 2... if it doesn't open
cap = cv2.VideoCapture(1)

# Optional: set resolution (C270 often supports 640x480 well)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing the device index (0/1/2...).")

fps = 0.0    

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = trt_model(frame)
    t1 = time.time()

    infer_fps = 1.0 / (t1 - t0)
    # Smooth FPS display (optional but recommended)
    fps = 0.9 * fps + 0.1 * infer_fps    
    print('FPS', fps)

    # Annotated visualization frame (BGR)
    annotated = results[0].plot()

    cv2.imshow("Webcam-YOLO", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()