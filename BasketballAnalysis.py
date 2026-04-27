from ultralytics import YOLO
import cv2
import math
import csv

cap = cv2.VideoCapture("vid (1).mp4")
model = YOLO("yolo11m.pt")

fps = cap.get(cv2.CAP_PROP_FPS)

prev_x = None
prev_y = None
frame = 0
last_make_frame = -9999
make_cooldown_frames = 10
rows = []
made_any_shot = False

while True:
    success, img = cap.read()
    frame+=1

    if not success:
        break

    results = model(img)

    annotated_frame = results[0].plot()
    
    cv2.putText(annotated_frame, f"FPS: {fps}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    rim_x = 370
    rim_y = 600

    rim_radius = 75

    cv2.circle(annotated_frame, (rim_x,rim_y), rim_radius, (0,255,0),thickness=3)

    speed = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        print(name)

        x,y, w,h = box.xywh[0]
        ball_x = int(x)
        ball_y = int(y)


        if name == "sports ball":

            if prev_x is not None and prev_y is not None:
                vx = (ball_x-prev_x)*fps
                vy = (ball_y-prev_y)*fps
                speed = math.sqrt(vx**2 + vy**2)
                
            else:
                vx = 0
                vy = 0
                speed =0

            dx = ball_x - rim_x
            dy = ball_y - rim_y

            distance = math.sqrt(dx**2 + dy**2)

            cv2.putText(annotated_frame, f"Dist: {int(distance)}", (ball_x,ball_y-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(annotated_frame, f"Speed: {int(speed)}", (ball_x, ball_y+30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 2)
            
            crosses_rim_plane = (
                prev_y is not None
                and prev_y < rim_y
                and ball_y >= rim_y
                and vy > 0
            )
            within_hoop_width = abs(dx) <= int(rim_radius * 0.75)
            cooldown_over = (frame - last_make_frame) > make_cooldown_frames

            if crosses_rim_plane and within_hoop_width and cooldown_over:
                label = 1
                last_make_frame = frame
                made_any_shot = True
            else:
                label = 0

            row = [frame, ball_x, ball_y, dx, dy, distance, vx, vy, speed, label]
            rows.append(row)

            prev_x = ball_x
            prev_y = ball_y

        
    cv2.imshow("img", annotated_frame)

    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if made_any_shot == True:
    final_label = 1
else:
    final_label = 0

with open("Final_Shots.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["frame", "ball_x", "ball_y", "dx", "dy", "distance", "vx", "vy", "speed", "label"])
    for row in rows:
        row[-1] = final_label
        writer.writerow(row)

    





