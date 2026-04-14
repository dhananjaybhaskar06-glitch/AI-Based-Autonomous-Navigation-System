import cv2
import numpy as np

# ================= YOLO SETUP =================
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= DISTANCE FUNCTION =================
def estimate_distance(box_height):
    if box_height > 300:
        return "VERY CLOSE"
    elif box_height > 150:
        return "MEDIUM"
    else:
        return "FAR"

# ================= LANE DETECTION =================
def detect_lanes_and_direction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height*0.6)),
        (0, int(height*0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50,
                            minLineLength=50, maxLineGap=50)

    line_image = np.zeros_like(frame)

    left_x = []
    right_x = []

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 5)

            if x1 < width//2 and x2 < width//2:
                left_x.extend([x1, x2])
            elif x1 > width//2 and x2 > width//2:
                right_x.extend([x1, x2])

    direction = "STRAIGHT"

    if len(left_x) > 0 and len(right_x) > 0:
        lane_center = (np.mean(left_x) + np.mean(right_x)) / 2
        frame_center = width / 2

        if lane_center < frame_center - 50:
            direction = "LEFT"
        elif lane_center > frame_center + 50:
            direction = "RIGHT"

    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return combined, direction

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    height, width, _ = frame.shape

    # -------- LANE DETECTION --------
    frame, direction = detect_lanes_and_direction(frame)

    # -------- OBJECT DETECTION --------
    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    # -------- DECISION + PATH PLANNING --------
    movement = "GO STRAIGHT"
    obstacle_center = False

    if len(indexes) > 0:
        for i in indexes:
            i = i[0] if isinstance(i, (list, tuple)) else i

            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            distance = estimate_distance(h)

            center_x_obj = x + w//2

            # Check if obstacle is in center
            if abs(center_x_obj - width//2) < 100:
                obstacle_center = True

            # DECISION
            if label == "person":
                decision = "STOP"
                print("🚫 STOP! Person detected -", distance)
            elif label in ["car", "truck", "bus"]:
                decision = "SLOW"
                print("⚠️ Vehicle nearby -", distance)
            else:
                decision = "CLEAR"

            # COLOR
            if decision == "STOP":
                color = (0,0,255)
            elif decision == "SLOW":
                color = (0,255,255)
            else:
                color = (0,255,0)

            # DRAW BOX
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

            text = f"{label} | {decision} | {distance}"
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # -------- PATH PLANNING --------
            if decision == "STOP":
                movement = "STOP"

            elif decision == "SLOW" and obstacle_center:
                if direction == "LEFT":
                    movement = "MOVE LEFT"
                elif direction == "RIGHT":
                    movement = "MOVE RIGHT"
                else:
                    movement = "TURN LEFT (AVOID)"

    # -------- FINAL MOVEMENT DISPLAY --------
    move_color = (0,255,0)

    if movement == "STOP":
        move_color = (0,0,255)
    elif "LEFT" in movement or "RIGHT" in movement:
        move_color = (255,0,0)

    cv2.putText(frame, f"MOVE: {movement}",
                (50,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                move_color,
                3)

    cv2.putText(frame, f"LANE: {direction}",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2)

    # -------- DISPLAY --------
    cv2.imshow("AI Navigation System (FINAL)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()