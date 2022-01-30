# python package
from keras.models import load_model
from keras.models import model_from_json

# internal package
from src.infra import infra

# Initialize Global alias
_appendListElement              = infra._appendListElement
_getSplitedStringByIndex        = infra._getSplitedStringByIndex
_getElementByIndex              = infra._getElementByIndex
_getFilesFromFolder             = infra._getFilesFromFolder
_getFilePathAndName             = infra._getFilePathAndName
_openImageFile                  = infra._openImageFile
_imageToNumpyArray              = infra._imageToNumpyArray
_renderRGBImage                 = infra._renderRGBImage
_renderRGBtoGrayImage           = infra._renderRGBtoGrayImage
_resizeImageByModelInputShape   = infra._resizeImageByModelInputShape
_normalizeImage                 = infra._normalizeImage
_reshapeGrayImage               = infra._reshapeGrayImage
_expandRGBImageDimensions       = infra._expandRGBImageDimensions

def _buildDictModel(list_model) -> list:
  keys         = []
  values       = []
  
  for model in list_model:

    if type(model) == list: # for json model (include json model and h5 weight)
      getModelPath      = _getElementByIndex(model, 0)
      getModelAndExt    = _getSplitedStringByIndex(getModelPath, "/", -1)
      getModelName      = _getSplitedStringByIndex(getModelAndExt, ".", 0)
      _appendListElement(keys, getModelName)
      _appendListElement(values, model)

    else:
      getModelAndExt    = _getSplitedStringByIndex(model, "/", -1)
      getModelName      = _getSplitedStringByIndex(getModelAndExt, ".", 0)
      _appendListElement(keys, getModelName)
      _appendListElement(values, model)

  return keys, values

def _buildListModel(path):
  json_arch       = []
  json_weight     = []
  hdf5_model      = []
  json_model      = []
  files_in_folder = _getFilesFromFolder(path)

  for data in files_in_folder:

    if "model.json" in data:
      file_name_and_path = _getFilePathAndName(path, data)
      _appendListElement(json_arch, file_name_and_path)

    elif "weights.h5" in data or "weights.hdf5" in data:
      file_name_and_path = _getFilePathAndName(path, data)
      _appendListElement(json_weight, file_name_and_path)

    elif "model.h5" in data or "model.hdf5" in data:
      file_name_and_path = _getFilePathAndName(path, data)
      _appendListElement(hdf5_model, file_name_and_path)

  if json_arch and json_weight:
    json_model = _getJsonModel(json_arch, json_weight)
  
  return json_model, hdf5_model
  
def _getDictModel(path):
  dicts        = {}
  keys         = []
  values       = []
  json_model, hdf5_model = _buildListModel(path)
  
  if json_model:  
    keys, values = _buildDictModel(json_model)

  if hdf5_model:
    keys, values = _buildDictModel(hdf5_model)
    
  for i in range(len(keys)):
    data        = _getElementByIndex(keys, i)
    dicts[data] = values[i]
    
  return dicts, keys, values

def _loadSelectModel(model, path):
  model_dict, _, _ = _getDictModel(path)

  for data in model_dict:

    if data == model:
      model_and_weight  = _getElementByIndex(model_dict, data)

      if type(model_and_weight) == list and model_and_weight:
        model_name        = _getElementByIndex(model_and_weight, 0)
        weight_name       = _getElementByIndex(model_and_weight, 1)
        json_file         = open(model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model      = model_from_json(loaded_model_json)
        loaded_model.load_weights(weight_name)

      else:
        loaded_model      = load_model(model_and_weight)

  return loaded_model
    
def _loadCompareModel(list_model, path):
  model_dict, _, _ = _getDictModel(path)
  list_ofModel = []

  for data in list_model:
    model_and_weight  = _getElementByIndex(model_dict, data)

    if data in model_dict:
      if type(model_and_weight) == list and model_and_weight:
        model_name        = _getElementByIndex(model_and_weight, 0)
        weight_name       = _getElementByIndex(model_and_weight, 1)
        json_file         = open(model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model      = model_from_json(loaded_model_json)
        loaded_model.load_weights(weight_name)
        _appendListElement(list_ofModel, loaded_model)

      else:
        loaded_model = load_model(model_and_weight)
        _appendListElement(list_ofModel, loaded_model)

  return list_ofModel

def _getJsonModel(models, weights)-> list:
  sub_value   = []
  data        = []
  models.sort(reverse=True)
  weights.sort()

  for weight in weights:
    model           = models.pop()
    getModelAndExt  = _getSplitedStringByIndex(model, "/", -1) # example result: VGG19_model.json
    modelName       = _getSplitedStringByIndex(getModelAndExt, "_", 0) # example result: VGG19
    if modelName in weight:
      _appendListElement(sub_value, model)
      _appendListElement(sub_value, weight)
      _appendListElement(data, sub_value)
    sub_value = []

  return data

def _rgbImageProcessing(imageFile, model):
  readImage           = _openImageFile(imageFile)
  imageNdarray        = _imageToNumpyArray(readImage)
  convertToRGB        = _renderRGBImage(imageNdarray)
  resizeImage ,_ ,_   = _resizeImageByModelInputShape(convertToRGB, model)
  normalizeImage      = _normalizeImage(resizeImage)
  resultImage         = _expandRGBImageDimensions(normalizeImage, 0)
  return resultImage

def _grayImageProcessing(imageFile, model):
  readImage                    = _openImageFile(imageFile)
  imageNdarray                 = _imageToNumpyArray(readImage)
  convertToRGB                 = _renderRGBImage(imageNdarray)
  convertToGray                = _renderRGBtoGrayImage(convertToRGB)
  resizeImage ,_ ,image_size   = _resizeImageByModelInputShape(convertToGray, model)
  normalizeImage               = _normalizeImage(resizeImage)
  resultImage                  = _reshapeGrayImage(normalizeImage, image_size)

  return resultImage