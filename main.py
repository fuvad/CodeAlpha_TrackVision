import cv2
from ultralytics import YOLO
from tracker import Tracker


model = YOLO("yolov8n.pt")

tracker = Tracker()


# webcam
cap = cv2.VideoCapture(0)

# if webcam not found
if not cap.isOpened():
    print("Cannot open webcam")
    exit()


with open("coco.txt", "r") as file:
    class_list = file.read().split("\n")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    detections = []

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            if confidence < 0.5:
                continue

            detections.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                    class_id,
                ]
            )

    tracked_input = [
        det[:4]
        for det in detections
    ]

    tracked_boxes = tracker.update(
        tracked_input
    )

    for tracked_box in tracked_boxes:

        x1, y1, x2, y2, obj_id = (
            tracked_box
        )

        label = "object"

        for det in detections:

            dx1, dy1, dx2, dy2, cid = det

            if (
                abs(x1 - dx1) < 10
                and abs(y1 - dy1) < 10
            ):
                label = class_list[cid]
                break

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"{label} ID:{obj_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imshow(
        "Object Detection + Tracking",
        frame,
    )

    # keep window alive
    key = cv2.waitKey(1)

    if key == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()