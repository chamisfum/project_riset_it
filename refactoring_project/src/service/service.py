# internal package
from src.config import config
from src.infra import infra

# 
BuildDictModel        = config._buildDictModel
BuildListModel        = config._buildListModel
GetDictModel          = config._getDictModel
LoadSelectModel       = config._loadSelectModel
LoadCompareModel      = config._loadCompareModel
GrayImageProcessing   = infra._grayImageProcessing
RGBImageProcessing    = infra._rgbImageProcessing
CurrentTime           = infra._getCurrentTime
DifferentTime         = infra._getDifferentTime
GetCollectionFiles    = infra._getFilesFromFolder
GetFilePathWithName   = infra._getFilePathAndName
MakePrediction        = infra._predictData
AppendListElement     = infra._appendListElement
RoundedListValue      = infra._roundedPercentageListValue
GetSplitedDataByIndex = infra._getSplitedStringByIndex

def GetListOfQueryImage(path):
  listClass = []
  listImage = []
  listQuery = []
  fileCollection = GetCollectionFiles(path)

  for data in fileCollection:
    AppendListElement(listImage, data)
    fullFilePath  = GetFilePathWithName(path, data)
    AppendListElement(listQuery, fullFilePath)
    classFile     = GetSplitedDataByIndex(data, "_", 0)
    AppendListElement(listClass, classFile)
    
  return listClass, listImage, listQuery

def PredictInputRGBImage(choosen_model, model_path, image):
  model             = LoadSelectModel(choosen_model, model_path)
  image             = RGBImageProcessing(image, model)
  start             = CurrentTime
  prediction        = MakePrediction(model, image)
  predictionTime    = DifferentTime(start)
  predictionRounded = RoundedListValue(prediction, 3)
  return predictionRounded, predictionTime

def PredictInputGrayImage(choosen_model, model_path, image):
  model             = LoadSelectModel(choosen_model, model_path)
  image             = GrayImageProcessing(image, model)
  start             = CurrentTime
  prediction        = MakePrediction(model, image)
  predictionTime    = DifferentTime(start)
  predictionRounded = RoundedListValue(prediction, 3)
  return predictionRounded, predictionTime

def PredictInputRGBImageList(list_choosen_model, model_path, image):
  predictionList    = []
  predictionTime    = []
  listOfLoadedModel = LoadCompareModel(list_choosen_model, model_path)

  for model in listOfLoadedModel:
    image             = RGBImageProcessing(image, model)
    start             = CurrentTime
    prediction        = MakePrediction(model, image)
    differentTime     = DifferentTime(start)
    AppendListElement(predictionTime, differentTime)
    predictionRounded = RoundedListValue(prediction, 3)
    AppendListElement(predictionList, predictionRounded)

  return predictionList, predictionTime

def PredictInputGrayImageList(list_choosen_model, model_path, image):
  predictionList    = []
  predictionTime    = []
  listOfLoadedModel = LoadCompareModel(list_choosen_model, model_path)

  for model in listOfLoadedModel:
    image             = GrayImageProcessing(image, model)
    start             = CurrentTime
    prediction        = MakePrediction(model, image)
    differentTime     = DifferentTime(start)
    AppendListElement(predictionTime, differentTime)
    predictionRounded = RoundedListValue(prediction, 3)
    AppendListElement(predictionList, predictionRounded)

  return predictionList, predictionTime
