
"""
DOCUMENTATION:
Infra layer use to provide a tiny and small function to handle our application
You can add new function features to handle a small acctivities as private func
Here is only use one rule (One Functinon One Acctivity)
"""

# python package
import cv2
import numpy as np
import os
from PIL import Image
import time

def _getFilesFromFolder(path) -> list:
    """
        _getFilesFromFolder : provide us a collection of file name with each path
        accept path location or folder that want to scan all data in those path
        
        EXAMPLE ARGS (path = '/usr/name/')
        
        EXAMPLE PROSSIBLE RESULT : ['image.jpg','image2.jpg']
    """
    list_files = os.listdir(path)
    return list_files

def _getFilePathAndName(path, file) -> str:
    """
        _getFilePathAndName : provide  string of file name and path location
        accept path folder or location of the file and file name as arguments 
        
        EXAMPLE ARGS : (path = '/usr/name/', file = 'image.jpg')
       
        EXAMPLE PROSSIBLE RESULT : '/usr/name/image.jpg'
    """
    file_path = os.path.join(path, file)
    return file_path
    
def _getDifferentTime(startTime) -> float:
    """
        _getDifferentTime : provide a difference of start time and current time 
        commonly used for measure how long each function activities running 
        accept startTime as beginning time and return a float number of difference 
        time with 4 digit decimal carracters.

        EXAMPLE ARGS : (startTime = time.time())
        
        EXAMPLE PROSSIBLE RESULT : (150222.102489129358)
    """
    current_time    = time.time()
    different_time  = current_time - startTime
    rounded_time    = _roundFloatNumber(different_time, 4)
    return rounded_time

def _predictData(model, file) -> list:
    """
        _predictData : provide a collection of prediction result
        accept model and file name + location as arguments

        EXAMPLE ARGS : (model = <keras.model>, file = '/usr/name/image.jpg')
        
        EXAMPLE PROSSIBLE RESULT : [85.33, 10,77, 3.0] etc
    """
    prediction = model.predict(file)[0]
    return prediction

def _roundFloatNumber(data, decimal_length) -> float:
    """
        _roundFloatNumber : provide float value with n decimal digit number
        accept data tobe rounded number and decimal_length as the number of 
        decimal digit number

        EXAMPLE ARGS : (data = 2.43527, decimal_length = 3)

        EXAMPLE PROSSIBLE RESULT : 2.435
    """
    res = round(data, decimal_length)
    return res

def _roundedPercentage(data, decimal_length) -> float:
    """
        _roundFloatNumber : provide a rounded float percentile value with n decimal
        digit number accept data as argument that would be rounded number and 
        decimal_length as the number of decimal digit number

        EXAMPLE ARGS : (data = 0.43527, decimal_length = 2)

        EXAMPLE PROSSIBLE RESULT : 43.53
    """
    percentage  = data * 100
    res         = _roundFloatNumber(percentage, decimal_length)
    return res

def _appendListElement(list_data, data) -> list:
    """
        _appendListElement : append new squence element and return into sequence
        accept list_data as list that would be append with new data. And the data 
        is arg to hold the new data

        EXAMPLE ARGS : (list_data = [], data = 2)

        EXAMPLE PROSSIBLE RESULT : [2]
    """
    list_data.append(data)
    return list_data

def _roundedPercentageListValue(list_data, decimal_length) -> list:
    """
        _roundedPercentageListValue : provide a collection of rounded percentile 
        value with n decimal digit number accept list_data as collection tobe rounded 
        percentile and decimal_length as the number of decimal digit number

        EXAMPLE ARGS : (data = [0.43527, 0.56593], decimal_length = 2)

        EXAMPLE PROSSIBLE RESULT : [43.53, 56.59]
    """
    res    = []
    for element in list_data:
        data    = _roundedPercentage(element, decimal_length)
        res     = _appendListElement(res, data)
    return res

def _splitDataByRegex(string_data, regex) -> list:
    """
        _splitDataByRegex : provide a collection of splitted string by regex
        accept string_data as a string that would be split into several part. 
        And regex value as splitter

        EXAMPLE ARGS : (string_data = "this_dataset", regex = "_")

        EXAMPLE PROSSIBLE RESULT : ["this", "dataset"]
    """
    res = string_data.split(regex)
    return res


def _getElementByIndex(list_data, index):
    """
        _getElementByIndex : provide a value of a collection data that indicate by defined index
        accept list_data as a collection data and index as the index of the choosen data that would be found

        EXAMPLE ARGS : (list_data = [0.43527, 0.56593], index = 0)

        EXAMPLE PROSSIBLE RESULT : 0.43527
    """
    res = list_data[index]
    return res

def _getElementByIndexRange(list_data, buttom=0, top=-1):
    """
        _getElementByIndex : might be provide a collection value of a collection data that indicate 
        by defined index range accept list_data as a collection data, buttom as index buttom of index 
        range (defaul = 0) and top as the top index range (default = -1)
        
        EXAMPLE ARGS : (list_data = [0.43527, 0.56593, 0.43527], buttom = 0, top = 1)

        EXAMPLE PROSSIBLE RESULT : [0.43527, 0.56593]
    """
    res = list_data[buttom:top]
    return res

def _getSplitedStringByIndex(string_data, regex, index) -> str:
    """
        _getSplitedStringByIndex : provide a choosen data of splitted string by regex with choosen 
        index of output accept string_data as a string that would be split into several part, 
        regex value as splitter, and index as indicator of choosen data

        EXAMPLE ARGS : (string_data = "this_dataset", regex = "_")

        EXAMPLE PROSSIBLE RESULT : ["this", "dataset"]
    """
    splited_string  = _splitDataByRegex(string_data, regex)
    res             = _getElementByIndex(splited_string, index)
    return res

def _openImageFile(image_file):
    """
        _openImageFile : used to open image file using pillow function

        EXAMPLE ARGS : (image_file = "/usr/name/image.jpg")

        EXAMPLE PROSSIBLE RESULT : <image.Metadata>
    """
    read_image      = Image.open(image_file)
    return read_image

def _imageToNumpyArray(image):
    """
        _imageToNumpyArray : transform image into numpy array

        EXAMPLE ARGS : (image = <image.Metadata>)

        EXAMPLE PROSSIBLE RESULT : <type:ndarray>
    """
    img_to_ndarray   = np.array(image)
    return img_to_ndarray

def _renderRGBImage(bgr_image):
    """
        _renderRGBImage : transform from BGR format into RGB format
        accept numpy array of image and return numpy array of image
    """
    rendered_image  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rendered_image

def _renderRGBtoGrayImage(rgb_image):
    """
        _renderRGBtoGrayImage : transform from BGR/RGB format into Grayscale format
        accept numpy array of image and return numpy array of image
        NOTE: commonly its use RGB format as input
    """
    rendered_image  = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    return rendered_image

def _getImageSizeFromModel(model, index=0, buttom=0, top=-1):
    """
        _getImageSizeFromModel : get image size and input shape dimention of model
        accept model and index as argument and return a model input shape with image size based on model input shape
    """
    model_input_shape   = _getElementByIndex(model.layers, index)
    image_size          = _getElementByIndexRange(model_input_shape.input_shape, buttom, top)
    return model_input_shape, image_size

def _resizeImageByModelInputShape(image, model):
    """
        _resizeImageByModelInputShape : resize image by model input shape
        accept model and image as argument and return resized image with input_shape and image_size
    """
    index                   = 0
    buttom_index            = 1
    top_index               = 3
    input_shape, image_size = _getImageSizeFromModel(model, index, buttom_index, top_index)
    resized_image           = cv2.resize(image, image_size)
    return resized_image, input_shape, image_size

def _normalizeImage(image):
    """
        _normalizeImage : normalize image file into float32 range value and devided by 255 (as pixel representation) 
        accept image as argument and return normalized_image
    """
    normalized_image   = image.astype('float32') / 255
    return normalized_image

def _reshapeGrayImage(image, image_size, gray_channel=(1,)):  
    """
        _reshapeGrayImage : reshaping an image into grayschale image dimension. 
        accept image, image_size and gray_channel scale as argument and return reshaped image
    """
    res = np.reshape(image, image_size + gray_channel)
    return res

def _expandRGBImageDimensions(image, axis=0): 
    """
        _expandRGBImageDimensions : expand RGB image dimension in addition to fit with model input shape
        accept image, and axis with default 0 value and return expanded image dimension
    """ 
    res = np.expand_dims(image, axis)
    return res
