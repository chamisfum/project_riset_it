# python package
from tkinter import image_names
import cv2
import numpy as np
import os
from PIL import Image
import time

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

def _getElementByIndex(list_data, index):
    res = list_data[index]
    return res

def _getElementByIndexRange(list_data, buttom=0, top=-1):
    res = list_data[buttom:top]
    return res

def _getSplitedStringByIndex(string_data, regex, index) -> str:
    splited_string  = _splitDataByRegex(string_data, regex)
    res             = _getElementByIndex(splited_string, index)
    return res

def _listLength(list_data) -> int:
    res = len(list_data)
    return

def _openImageFile(image_file):
    read_image      = Image.open(image_file)
    return read_image

def _imageToNumpyArray(image):
    img_to_ndarray   = np.array(image)
    return img_to_ndarray

def _renderRGBImage(bgr_image):
    rendered_image  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rendered_image

def _renderRGBtoGrayImage(rgb_image):
    rendered_image  = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    return rendered_image

def _getImageSizeFromModel(model, index=0, buttom=0, top=-1):
    model_input_shape   = _getElementByIndex(model.layers, index)
    image_size          = _getElementByIndexRange(model_input_shape.input_shape, buttom, top)
    return model_input_shape, image_size

def _resizeImageByModelInputShape(image, model):
    index                   = 0
    buttom_index            = 1
    top_index               = 3
    input_shape, image_size = _getImageSizeFromModel(model, index, buttom_index, top_index)
    resized_image           = cv2.resize(image, image_size)
    return resized_image, input_shape, image_size

def _normalizeImage(image):
    normalized_image   = image.astype('float32') / 255
    return normalized_image

def _reshapeGrayImage(image, image_size, gray_channel=(1,)):  
    res = np.reshape(image, image_size + gray_channel)
    return res

def _expandRGBImageDimensions(image, axis=0):  
    res = np.expand_dims(image, axis)
    return res
