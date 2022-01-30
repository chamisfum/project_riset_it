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
GetDictModel               = config._getDictModel

_differentTime             = infra._getDifferentTime
_getCollectionFiles        = infra._getFilesFromFolder
_getFilePathWithName       = infra._getFilePathAndName
_makePrediction            = infra._predictData
_appendListElement         = infra._appendListElement
_roundedListValue          = infra._roundedPercentageListValue
_getSplitedDataByIndex     = infra._getSplitedStringByIndex

def GetListOfQueryImage(path):
  listClass = []
  listImage = []
  listQuery = []
  fileCollection = _getCollectionFiles(path)

  for data in fileCollection:
    _appendListElement(listImage, data)
    fullFilePath  = _getFilePathWithName(path, data)
    _appendListElement(listQuery, fullFilePath)
    classFile     = _getSplitedDataByIndex(data, "_", 0)
    _appendListElement(listClass, classFile)
    
  return listClass, listImage, listQuery

def PredictInputRGBImage(choosen_model, model_path, image):
  model             = _loadSelectModel(choosen_model, model_path)
  image_data        = _rgbImageProcessing(image, model)
  start             = time.time()
  prediction        = _makePrediction(model, image_data)
  predictionTime    = _differentTime(start)
  predictionRounded = _roundedListValue(prediction, 3)
  return predictionRounded, predictionTime

def PredictInputGrayImage(choosen_model, model_path, image):
  model             = _loadSelectModel(choosen_model, model_path)
  image_data        = _grayImageProcessing(image, model)
  start             = time.time()
  prediction        = _makePrediction(model, image_data)
  predictionTime    = _differentTime(start)
  predictionRounded = _roundedListValue(prediction, 3)
  return predictionRounded, predictionTime

def PredictInputRGBImageList(list_choosen_model, model_path, image):
  predictionList    = []
  predictionTime    = []
  listOfLoadedModel = _loadCompareModel(list_choosen_model, model_path)

  for model in listOfLoadedModel:
    image_data        = _rgbImageProcessing(image, model)
    start             = time.time()
    prediction        = _makePrediction(model, image_data)
    differentTime     = _differentTime(start)
    _appendListElement(predictionTime, differentTime)
    predictionRounded = _roundedListValue(prediction, 3)
    _appendListElement(predictionList, predictionRounded)

  return predictionList, predictionTime

def PredictInputGrayImageList(list_choosen_model, model_path, image):
  predictionList    = []
  predictionTime    = []
  listOfLoadedModel = _loadCompareModel(list_choosen_model, model_path)

  for model in listOfLoadedModel:
    image_data        = _grayImageProcessing(image, model)
    start             = time.time()
    prediction        = _makePrediction(model, image_data)
    differentTime     = _differentTime(start)
    _appendListElement(predictionTime, differentTime)
    predictionRounded = _roundedListValue(prediction, 3)
    _appendListElement(predictionList, predictionRounded)

  return predictionList, predictionTime
