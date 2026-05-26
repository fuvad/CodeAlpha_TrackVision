import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# Load YOLO model
model = YOLO("yolov8n.pt")


# Initialize Deep SORT tracker
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.4,
)


# Webcam
cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

# Video file example:
# cap = cv2.VideoCapture(
#     "videos/sample.mp4"
# )


# Check webcam
if not cap.isOpened():
    print("Cannot open webcam")
    exit()


# Load COCO classes safely
with open(
    "coco.txt",
    "r",
    encoding="utf-8"
) as file:

    class_list = [
        line.strip()
        for line in file
        if line.strip()
    ]


while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break


    # Run YOLO
    results = model(frame)

    detections = []


    for result in results:

        boxes = result.boxes


        for box in boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )


            # Ignore weak detections
            if confidence < 0.5:
                continue


            # Safe label lookup
            if class_id < len(
                class_list
            ):
                label = class_list[
                    class_id
                ]
            else:
                label = "unknown"


            # Deep SORT format:
            # ([left, top, width, height],
            # confidence,
            # class_name)

            detections.append(
                (
                    [
                        int(x1),
                        int(y1),
                        int(x2 - x1),
                        int(y2 - y1),
                    ],
                    confidence,
                    label,
                )
            )


    # Update tracker
    tracks = tracker.update_tracks(
        detections,
        frame=frame,
    )


    # Draw tracked objects
    for track in tracks:

        if not track.is_confirmed():
            continue


        track_id = (
            track.track_id
        )

        label = (
            track.get_det_class()
        )


        x1, y1, x2, y2 = map(
            int,
            track.to_ltrb()
        )


        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )


        # Label + ID
        cv2.putText(
            frame,
            f"{label} ID:{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )


    cv2.imshow(
        "TrackVision - YOLOv8 + Deep SORT",
        frame,
    )


    # Quit
    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):
        break


cap.release()

cv2.destroyAllWindows()