"""

DOCUMENTATION:

service layer use to provide a data processing that needed by user request. This layer can be used as 
service provider for application layer to provide several data. This layer would use several config 
backend as main helper to build each service.

"""

# python package
import time

# internal package
from src.config import config
from src.infra import infra

# Initialize Global alias
_loadSelectModel           = config._loadSelectModel
_loadCompareModel          = config._loadCompareModel
_grayImageProcessing       = config._grayImageProcessing
_rgbImageProcessing        = config._rgbImageProcessing
_getDictModel              = config._getDictModel

_differentTime             = infra._getDifferentTime
_getCollectionFiles        = infra._getFilesFromFolder
_getFilePathWithName       = infra._getFilePathAndName
_makePrediction            = infra._predictData
_appendListElement         = infra._appendListElement
_roundedListValue          = infra._roundedPercentageListValue
_getSplitedDataByIndex     = infra._getSplitedStringByIndex

def GetListOfQueryImage(path):
  """
  GetListOfQueryImage() : Provide a tuple of collection such (class name, image path, and image name)
                          This function will scan all files that contain in path folder and extract some information such (class name, image path, and image name) 
                          
                          ACCEPT path location of image file as argument

                          RETURN listClass, listImage, and listQuery

                          RETURN EXAMPLE :
                                 
                                 * LISTCLASS : ['Glioma', 'Meningioma','Meningioma', 'Pituitary', 'Pituitary', 'Pituitary']
                                 
                                 * LISTCLASS : ['Glioma_1469.png', 'Meningioma_09965.png', 'Meningioma_1.jpg',
                                                'Pituitary_11710.png', 'Pituitary_12556.png', 'Pituitary_13472.png']
                                 
                                 * LISTCLASS : ['static/queryImage/Glioma_1469.png', 'static/queryImage/Meningioma_09965.png', 
                                                'static/queryImage/Meningioma_1.jpg', 'static/queryImage/Pituitary_11710.png',
                                                'static/queryImage/Pituitary_12556.png', 'static/queryImage/Pituitary_13472.png',]
  """
  listClass = []
  listImage = []
  listQuery = []
  fileCollection = _getCollectionFiles(path)

  for data in fileCollection:
    classFile     = _getSplitedDataByIndex(data, "_", 0)
    _appendListElement(listClass, classFile)
    _appendListElement(listImage, data)
    fullFilePath  = _getFilePathWithName(path, data)
    _appendListElement(listQuery, fullFilePath)
  return listClass, listImage, listQuery

def PredictInputRGBImage(choosen_model, model_path, image):
  """
  PredictInputRGBImage() : Provide a tuple data which contain list of prediction result and how long prediction takes time
                          
                          This function would automatically scan choosen_model that contain in model_path directory and process 
                          RGB image before making a prediction.

                          This function is used to predict an RGB image. It can only process RGB image type since this function 
                          used _rgbImageProcessing() function to applied RGB image processing before making a prediction

                          ACCEPT choosen_model, model_path, input images as argument
                          
                          RETURN predictionResult, predictionTime

                          RETURN EXAMPLE :
                                 
                                 * predictionResult : is rounded of prediction result 
                                 -> [0.003, 99.987, 0.01]
                                 
                                 * predictionTime   : is prediction takes time 
                                 -> 0.1728
  """
  model             = _loadSelectModel(choosen_model, model_path)
  image_data        = _rgbImageProcessing(image, model)
  start             = time.time()
  prediction        = _makePrediction(model, image_data)
  predictionTime    = _differentTime(start)
  predictionResult  = _roundedListValue(prediction, 3)
  return predictionResult, predictionTime

def PredictInputGrayImage(choosen_model, model_path, image):
  """
  PredictInputGrayImage() : Provide a tuple data which contain list of prediction result and how long prediction takes time
                            
                          This function would automatically scan choosen_model that contain in model_path directory and process 
                          Grayscale image before making a prediction.
                          
                          This function is used to predict an Grayscale image. It can only process Grayscale image type since this function 
                          used _grayImageProcessing() function to applied Grayscale image processing before making a prediction
                          
                          ACCEPT choosen_model, model_path, input images as argument
                          
                          RETURN predictionResult, predictionTime

                          RETURN EXAMPLE :
                                 
                                 * predictionResult : is rounded of prediction result 
                                 -> [0.003, 99.987, 0.01]
                                 
                                 * predictionTime   : is prediction takes time 
                                 -> 0.1728
  """
  model             = _loadSelectModel(choosen_model, model_path)
  image_data        = _grayImageProcessing(image, model)
  start             = time.time()
  prediction        = _makePrediction(model, image_data)
  predictionTime    = _differentTime(start)
  predictionResult  = _roundedListValue(prediction, 3)

  return predictionResult, predictionTime

def PredictInputRGBImageList(list_choosen_model, model_path, image):
  """
  PredictInputRGBImageList() : Provide a tuple of collection data which contain prediction result and how long prediction takes time
                          
                          This function would automatically scan list choosen_model that contain in model_path directory and process 
                          RGB image before making a prediction.
                          
                          This function is would predict an RGB image with several selected model. It can only process RGB image type 
                          since this function used _rgbImageProcessing() to applied bounch of RGB image processing before making any prediction.

                          ACCEPT list_choosen_model, model_path, input images as argument
                          
                          RETURN predictionResult, predictionTime as list data

                          RETURN EXAMPLE :
                                 
                                 * predictionResult : are collection of rounded prediction result of each selected model 
                                                      -> [[0.003, 99.987, 0.01], [0.003, 99.987, 0.01]]
                                 
                                 * predictionTime   : is prediction takes time 
                                                      -> [0.1728, 0.1987]
  """
  predictionResult    = []
  predictionTime      = []
  listOfLoadedModel   = _loadCompareModel(list_choosen_model, model_path)

  for model in listOfLoadedModel:
    image_data        = _rgbImageProcessing(image, model)
    start             = time.time()
    prediction        = _makePrediction(model, image_data)
    differentTime     = _differentTime(start)
    _appendListElement(predictionTime, differentTime)
    predictionRounded = _roundedListValue(prediction, 3)
    _appendListElement(predictionResult, predictionRounded)

  return predictionResult, predictionTime

def PredictInputGrayImageList(list_choosen_model, model_path, image):
  """
  PredictInputGrayImageList() : Provide a tuple of collection data which contain prediction result and how long prediction takes time
                          
                          This function would automatically scan list choosen_model that contain in model_path directory and process 
                          Grayscale image before making a prediction.
                          
                          This function is would predict an Grayscale image with several selected model. It can only process Grayscale image type 
                          since this function used _grayImageProcessing() to applied bounch of Grayscale image processing before making any prediction.

                          ACCEPT list_choosen_model, model_path, input images as argument
                          
                          RETURN predictionResult, predictionTime as list data

                          RETURN EXAMPLE :
                                 
                                 * predictionResult : are collection of rounded prediction result of each selected model 
                                                      -> [[0.003, 99.987, 0.01], [0.003, 99.987, 0.01]]
                                 
                                 * predictionTime   : is prediction takes time -> [0.1728, 0.1987]
  """
  predictionResult  = []
  predictionTime    = []
  listOfLoadedModel = _loadCompareModel(list_choosen_model, model_path)

  for model in listOfLoadedModel:
    image_data        = _grayImageProcessing(image, model)
    start             = time.time()
    prediction        = _makePrediction(model, image_data)
    differentTime     = _differentTime(start)
    _appendListElement(predictionTime, differentTime)
    predictionRounded = _roundedListValue(prediction, 3)
    _appendListElement(predictionResult, predictionRounded)

  return predictionResult, predictionTime
