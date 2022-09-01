import torch
import pandas as pd
import numpy as np
import datetime

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

imgs = "./transito01.webp"

# Inference
#print()  # or .show(), .save()
results = model(imgs)
var = results.pandas().xyxy[0]
#print(type(results.pandas().xyxy[0]))

dataFrame = pd.DataFrame(var, columns = ["confidence", "name"])

results.save()

print(dataFrame)

# ser = pd.Series(dataFrame["name"])

# print(ser)


#pega a quantidade de elementos de um dataframe 
qtd = dataFrame[dataFrame.columns[0]].count()
print(qtd)


#transformando o Dataframe em um array
#valores = dataFrame.to_records()
#valores = dataFrame.reset_index()
#valores = dataFrame.values
# valores = dataFrame.to_numpy()



#transforma dataframe em json
# json = dataFrame.to_json()
# print(json)


#Trabalhando com data
# datetime.date.today()
# datetime.now()
# dataFrame['date'] = pd.to_datetime(dataFrame.date, format='%d-%m-%Y')
# dataFrame['date_DD_MM_YYYY'] = dataFrame['date'].dt.strftime('%d-%m-%Y')

