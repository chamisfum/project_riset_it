# internal package
from src.config import config
from src.infra import infra

# Initialize Global alias
_buildDictModel           = config._buildDictModel
_buildListModel           = config._buildListModel
_loadSelectModel          = config._loadSelectModel
_loadCompareModel         = config._loadCompareModel
_grayImageProcessing      = config._grayImageProcessing
_rgbImageProcessing       = config._rgbImageProcessing

_currentTime               = infra._getCurrentTime
_differentTime             = infra._getDifferentTime
_getCollectionFiles        = infra._getFilesFromFolder
_getFilePathWithName       = infra._getFilePathAndName
_makePrediction            = infra._predictData
_getElementByIndex         = infra._getElementByIndex
_listLength                = infra._listLength
_appendListElement         = infra._appendListElement
_roundedListValue          = infra._roundedPercentageListValue
_getSplitedDataByIndex     = infra._getSplitedStringByIndex
_getSplitedStringByIndex   = infra._getSplitedStringByIndex

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
  image             = _rgbImageProcessing(image, model)
  start             = _currentTime
  prediction        = _makePrediction(model, image)
  predictionTime    = _differentTime(start)
  predictionRounded = _roundedListValue(prediction, 3)
  return predictionRounded, predictionTime

def PredictInputGrayImage(choosen_model, model_path, image):
  model             = _loadSelectModel(choosen_model, model_path)
  image             = _grayImageProcessing(image, model)
  start             = _currentTime
  prediction        = _makePrediction(model, image)
  predictionTime    = _differentTime(start)
  predictionRounded = _roundedListValue(prediction, 3)
  return predictionRounded, predictionTime

def PredictInputRGBImageList(list_choosen_model, model_path, image):
  predictionList    = []
  predictionTime    = []
  listOfLoadedModel = _loadCompareModel(list_choosen_model, model_path)

  for model in listOfLoadedModel:
    image             = _rgbImageProcessing(image, model)
    start             = _currentTime
    prediction        = _makePrediction(model, image)
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
    image             = _grayImageProcessing(image, model)
    start             = _currentTime
    prediction        = _makePrediction(model, image)
    differentTime     = _differentTime(start)
    _appendListElement(predictionTime, differentTime)
    predictionRounded = _roundedListValue(prediction, 3)
    _appendListElement(predictionList, predictionRounded)

  return predictionList, predictionTime

def GetDictModel(path):
  dicts        = {}
  keys         = []
  values       = []
  json_model, hdf5_model = _buildListModel(path)
  
  if json_model:  
    key, value = _buildDictModel(json_model)
    _appendListElement(keys, key)
    _appendListElement(values, value)

  if hdf5_model:
    key, value = _buildDictModel(hdf5_model)
    _appendListElement(keys, key)
    _appendListElement(values, value)

  for i in range(_listLength(keys)):
    data              = _getElementByIndex(keys, i)
    model_name        = _getSplitedStringByIndex(data, ".", 0)
    dicts[model_name] = values[i]
    
  return dicts, keys, values
