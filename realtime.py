import cv2
import signal
import sys
import time
from ultralytics import YOLO

# Params
MODEL = "trained_models/best.onnx"
WEBCAM_INDEX = 0
CONFIDENCE = 0.25
IOU = 0.45
ENABLE_TRACKING = False
SELECTED_CLASSES = None
IMG_SIZE = 832
SHOW_FPS = True

cap = None

def signal_handler(sig, frame):
    global cap
    print("\nKeyboard interrupt detected. Cleaning up resources...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

def main():
    global cap
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Carregando o modelo: {MODEL}")
    model = YOLO(MODEL)
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    
    prev_time = 0
    curr_time = 0
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam.")
                break
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            if ENABLE_TRACKING:
                results = model.track(
                    frame, 
                    conf=CONFIDENCE, 
                    iou=IOU, 
                    classes=SELECTED_CLASSES,
                    imgsz=IMG_SIZE,
                    persist=True
                )
            else:
                results = model(
                    frame, 
                    conf=CONFIDENCE, 
                    iou=IOU, 
                    classes=SELECTED_CLASSES,
                    imgsz=IMG_SIZE
                )
            
            annotated_frame = results[0].plot()
            
            if SHOW_FPS:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(
                    annotated_frame, 
                    fps_text, 
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (0, 255, 0),
                    2 
                )
            
            cv2.imshow("Original", frame)
            cv2.imshow("Detected Objects", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()