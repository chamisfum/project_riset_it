# python package
from cgitb import reset
import cv2
import numpy as np
import os
from PIL import Image
import time

def _rgbImageProcessing(imageFile, model):
  readImage     = np.array(Image.open(imageFile))
  convertToRGB  = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
  resizeImage   = cv2.resize(convertToRGB, model.layers[0].input_shape[1:3])
  resultImage   = np.expand_dims(resizeImage.astype('float32') / 255, axis=0)

  return resultImage

def _grayImageProcessing(imageFile, model):
  readImage     = np.array(Image.open(imageFile))
  convertToRGB  = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
  convertToGray = cv2.cvtColor(convertToRGB, cv2.COLOR_BGR2GRAY)
  resizeImage   = cv2.resize(convertToGray, model.layers[0].input_shape[1:3]).astype('float32') / 255
  resultImage   = np.reshape(resizeImage, model.layers[0].input_shape[1:]+(1,))

  return resultImage

def _getCurrentTime() -> float:
    return round(time.time(), 4)

def _getFilesFromFolder(path) -> list:
    list_files = os.listdir(path)
    return list_files

def _getFilePathAndName(path, file) -> str:
    file_path = os.path.join(path, file)
    return file_path
    
def _getDifferentTime(startTime) -> float:
    different_time = time.time()-startTime
    rounded_result = _roundFloatNumber(different_time, 4)
    return rounded_result

def _predictData(model, file) -> list:
    prediction = model.predict(file)[0]
    return prediction

def _roundFloatNumber(data, decimal_length) -> float:
    res = round(data, decimal_length)
    return res

def _roundedPercentage(data, decimal_length) -> float:
    percentage  = data * 100
    res         = _roundFloatNumber(percentage, decimal_length)
    return res

def _appendListElement(list_data, data) -> list:
    list_data.append(data)
    return list_data

def _roundedPercentageListValue(list_data, decimal_length) -> list:
    res    = []
    for element in list_data:
        data    = _roundedPercentage(element, decimal_length)
        res     = _appendListElement(res, data)
    return res

def _splitDataByRegex(string_data, regex) -> list:
    res = string_data.split(regex)
    return res

def _getListElementByIndex(list_data, index):
    res = list_data[index]
    return res

def _getSplitedStringByIndex(string_data, regex, index) -> str:
    splited_string  = _splitDataByRegex(string_data, regex)
    res             = _getListElementByIndex(splited_string, index)
    return res

def _listLength(list_data) -> int:
    res = len(list_data)
    return

