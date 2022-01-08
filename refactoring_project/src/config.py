import cv2
import os
import time
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.models import model_from_json


def GetJsonModel(models, weights)-> list:
  sub_value   = []
  data        = []
  models.sort(reverse=True)
  weights.sort()

  for weight in weights:
    model = models.pop()
    if model.split("/")[-1].split("_")[0] in weight:
      sub_value.append(model)
      sub_value.append(weight)
      data.append(sub_value)
    sub_value = []

  return data
def GetListOfQueryImage(path):
  listClass = []
  listImage = []
  listQuery = []

  for data in os.listdir(path):
    listImage.append(data)
    listQuery.append(os.path.join(path, data))
    listClass.append(data.split("_")[0])
    
  return listClass, listImage, listQuery

def BuildListModel(path):
  json_arch    = []
  json_weight  = []
  hdf5_model   = []
  json_model   = []

  for data in os.listdir(path):
    if "model.json" in data:
      json_arch.append(os.path.join(path, data))
    elif "weights.h5" in data or "weights.hdf5" in data:
      json_weight.append(os.path.join(path, data))
    elif "model.h5" in data or "model.hdf5" in data:
      hdf5_model.append(os.path.join(path, data))
  if json_arch and json_weight:
    json_model = GetJsonModel(json_arch, json_weight)
  
  return json_model, hdf5_model

def BuildDictModel(list_model):
  keys         = []
  values       = []
  
  for model in list_model:
    if type(model) == list:
      keys.append(model[0].split("/")[-1].split(".")[0])
      values.append(model)
    else:
      keys.append(model.split("/")[-1].split(".")[0])
      values.append(model)

  return keys, values

def GetDictModel(path):
  dicts        = {}
  keys         = []
  values       = []
  json_model, hdf5_model = BuildListModel(path)

  if json_model:  
    key, value = BuildDictModel(json_model)
    keys, values   = keys + key, values + value
    
  if hdf5_model:
    key, value = BuildDictModel(hdf5_model)
    keys, values   = keys + key, values + value

  for i in range(len(keys)):
    dicts[keys[i].split(".")[0]] = values[i]

  return dicts, keys, values

def LoadSelectModel(model, path):
  model_dict, _, _ = GetDictModel(path)
  print(model_dict)
  for data in model_dict:
    if data == model:
      if type(model_dict[data]) == list and model_dict[data]:
        json_file = open(model_dict[data][0], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_dict[data][1])
      else:
        loaded_model = load_model(model_dict[data])
  return loaded_model
    
def LoadCompareModel(list_model, path):
  model_dict, _, _ = GetDictModel(path)
  list_ofModel = []
  for data in list_model:
    if data in model_dict:
      if type(model_dict[data]) == list and model_dict[data]:
        json_file = open(model_dict[data][0], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_dict[data][1])
        list_ofModel.append(loaded_model)
      else:
        loaded_model = load_model(model_dict[data])
        list_ofModel.append(loaded_model)
  return list_ofModel

def ImageProcessingRGB(imageFile, model):
  readImage     = np.array(Image.open(imageFile))
  convertToRGB  = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
  resizeImage   = cv2.resize(convertToRGB, model.layers[0].input_shape[1:3])
  resultImage   = np.expand_dims(resizeImage.astype('float32') / 255, axis=0)
  return resultImage

def ImageProcessingGray(imageFile, model):
  readImage     = np.array(Image.open(imageFile))
  convertToRGB  = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
  convertToGray = cv2.cvtColor(convertToRGB, cv2.COLOR_BGR2GRAY)
  resizeImage   = cv2.resize(convertToGray, model.layers[0].input_shape[1:3]).astype('float32') / 255
  resultImage   = np.reshape(resizeImage, model.layers[0].input_shape[1:]+(1,))
  return resultImage

def PredictInputImage(model, image):
  start             = time.time()
  prediction        = model.predict(image)[0]
  predictionTime    = round(time.time()-start,4)
  roundedPrediction = [round(elem * 100, 2) for elem in prediction]
  return roundedPrediction, predictionTime

def PredictInputRGBImageList(listModel, image):
  roundedPrediction = []
  predictionTime    = []
  for model in listModel:
    imageData         = ImageProcessingRGB(image, model)
    start             = time.time()
    prediction        = model.predict(imageData)[0]
    predictionTime.append(round(time.time()-start,4))
    roundedPrediction.append([round(elem * 100, 2) for elem in prediction])
  return roundedPrediction, predictionTime

def PredictInputGrayImageList(listModel, image):
  roundedPrediction = []
  predictionTime    = []
  for model in listModel:
    imageData         = ImageProcessingGray(image, model)
    start             = time.time()
    prediction        = model.predict(imageData)[0]
    predictionTime.append(round(time.time()-start,4))
    roundedPrediction.append([round(elem * 100, 2) for elem in prediction])
  return roundedPrediction, predictionTime