import pandas as pd
import torch 
import torch.nn as nn


df = pd.read_csv("shots_100.csv")

x = df.drop("made", axis = 1)
y = df['made']

x1 = torch.tensor(x.values, dtype = torch.float32)
y1 = torch.tensor(y.values, dtype = torch.float32).view(-1,1)


model = nn.Sequential(
    nn.Linear(5,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,1),
    
    nn.Sigmoid()

)

loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    prediction = model(x1)
    l = loss(prediction, y1)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, Loss: {l.item()}" )

with torch.no_grad():
    prediction = model(x1)
    print("Prediction", prediction)
    percentage_predictor = []
    new_predlist = []
    original_list = df['made'].tolist()
    for p in prediction:
        percentage_predictor.append(f"{(p.item()*100):.2f}")

        if p.item()>0.5:
            new_predlist.append(1)
        else:
            new_predlist.append(0)

    score = 0
    for i in range(len(new_predlist)):
        if original_list[i] == new_predlist[i]:
            score+=1
    
    accuracy = (score/len(original_list))*100
        

    print("Chance the shot goes in:",percentage_predictor)
    print("Original: ", original_list)
    print("New: ",new_predlist)
    
    print("Final accuracy is:", accuracy,"%")
  


