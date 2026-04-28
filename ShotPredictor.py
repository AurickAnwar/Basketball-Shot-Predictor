from ultralytics import YOLO
import cv2
import math
import csv
import os
import pandas as pd
import torch
import torch.nn as nn


df = pd.read_csv("Final_Shots.csv")

df = df[df["frame"] != "frame"]#Remove header rows that may have been appended to the CSV file
df =df.astype(float)
x = df.drop("made", axis = 1)
y = df['made']

x1 = torch.tensor(x.values, dtype = torch.float32)
y1 = torch.tensor(y.values, dtype = torch.float32).view(-1,1)


pytorch_model = nn.Sequential(#break down the layers of the model
    nn.Linear(9,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,1),
    
    nn.Sigmoid()

)

loss = nn.BCELoss()#To predict probability between 0 and 1
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

for epoch in range(1000):
    prediction = pytorch_model(x1)
    l = loss(prediction, y1)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, Loss: {l.item()}" )#print each epoch and loss to see how the model is learning

with torch.no_grad():#Make predictions with the trained model
    prediction = pytorch_model(x1)
    print("Prediction", prediction)
    percentage_predictor = []
    new_predlist = []
    original_list = df['made'].tolist()
    for p in prediction:
        percentage_predictor.append(f"{(p.item()*100):.2f}")

        if p.item()>0.5:
            new_predlist.append(1)#if shot chance is greater than 50% predict it as a make
        else:
            new_predlist.append(0)

    score = 0
    for i in range(len(new_predlist)):
        if original_list[i] == new_predlist[i]:
            score+=1
    
    accuracy = (score/len(original_list))*100


cap = cv2.VideoCapture("vid (2).mp4")
yolo_model = YOLO("yolo11m.pt")

fps = cap.get(cv2.CAP_PROP_FPS)

prev_x = None
prev_y = None
frame = 0
last_make_frame = -9999
make_cooldown_frames = 10
rows = []
made_any_shot = False
last_chance = 0

while True:
    success, img = cap.read()
    frame+=1

    if not success:
        break

    results = yolo_model(img)

    annotated_frame = results[0].plot()#draw bounding boxes on the frame
    
    cv2.putText(annotated_frame, f"FPS: {fps}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    rim_x = 370
    rim_y = 600

    rim_radius = 75

    cv2.circle(annotated_frame, (rim_x,rim_y), rim_radius, (0,255,0),thickness=3)#draw a circle around the rim to visualize the area where the ball needs to pass through to be considered a make

    speed = 0

    for box in results[0].boxes:#bounding box info for each detected object in the frame
        cls = int(box.cls[0])
        name = yolo_model.names[cls]
        

        x,y, w,h = box.xywh[0] #center coordinates and dimensions of the bounding box
        ball_x = int(x)
        ball_y = int(y)


        if name == "sports ball":#if the detected object is a sports ball

            if prev_x is not None and prev_y is not None:#object's velocity and speed if there is a previous position to compare to
                vx = (ball_x-prev_x)*fps
                vy = (ball_y-prev_y)*fps
                speed = math.sqrt(vx**2 + vy**2)
                
            else:
                vx = 0
                vy = 0
                speed =0

            dx = ball_x - rim_x #distance from the ball to the rim in both x and y directions
            dy = ball_y - rim_y

            distance = math.sqrt(dx**2 + dy**2) #total distance from the ball to the rim

            with torch.no_grad():
                input_tensor = torch.tensor([[frame, ball_x, ball_y, dx, dy, distance, vx, vy, speed]], dtype=torch.float32)#tensor with all the features of the current frame
                shot_chance = pytorch_model(input_tensor).item()#predict the chance that the shot will be made using the trained model
                last_chance = shot_chance
            cv2.putText(annotated_frame, f"Chance Shot is made: {last_chance*100:.2f}%", (20,95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(annotated_frame, f"Accuracy: {accuracy:.2f}%",(20,150),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)


            cv2.putText(annotated_frame, f"Dist: {int(distance)}", (ball_x,ball_y-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(annotated_frame, f"Speed: {int(speed)}", (ball_x, ball_y+30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 2)


            #calculation for if the ball is crossing the plane of the rim, if it is within the width of the hoop, and if enough frames have passed since the last detected make to avoid multiple detections for the same shot
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



#writing the collected data for each frame to a CSV file
file_exists = os.path.isfile("Final_Shots.csv") and os.path.getsize("Final_Shots.csv") > 0

with open("Final_Shots.csv", "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(["frame", "ball_x", "ball_y", "dx", "dy", "distance", "vx", "vy", "speed", "made"])

    for row in rows:
        row[-1] = final_label
        writer.writerow(row)






    





