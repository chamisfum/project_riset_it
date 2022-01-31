"""

DOCUMENTATION:

config layer use to provide several helper and configuration service that needed by service layer.
This layer can be used as config provider and controller for service layer to provide several function configuration.
This layer would use any infrastructure layer function to build each configuration services.

"""
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
  """
  _buildDictModel() : Provide a collection of model name and collection of model path
                      This function will help to build Model dictionary by providing each key and value
                      This function will handle both of json model or h5 model 
                      
                      ACCEPT list of model either json or h5 model as argument
                      
                      RETURN keys, values

                      RETURN EXAMPLE : 
                                 * KEYS   : ['BALANCE_model', 'IMBALANCE_model', 'SPLIT_AUGMENTATION_model']

                                 * VALUES : ['static/model/BALANCE_model.h5', 'static/model/IMBALANCE_model.h5', 
                                            'static/model/SPLIT_AUGMENTATION_model.h5']
  """
  keys         = []
  values       = []
  
  for model in list_model:

    if type(model) == list: # handle json model (include json model and h5 weight)
      getModelPath        = _getElementByIndex(model, 0)
      get_model_and_ext   = _getSplitedStringByIndex(getModelPath, "/", -1)
      get_model_name      = _getSplitedStringByIndex(get_model_and_ext, ".", 0)
      _appendListElement(keys, get_model_name)
      _appendListElement(values, model)

    else: # handle hdf5 or H5 model
      get_model_and_ext   = _getSplitedStringByIndex(model, "/", -1)
      get_model_name      = _getSplitedStringByIndex(get_model_and_ext, ".", 0)
      _appendListElement(keys, get_model_name)
      _appendListElement(values, model)

  return keys, values

def _buildListModel(path):
  """
  _buildListModel() : Provide a collection of model and weight path
                      This function will help to build Model dictionary by providing a collection model and weight path
                      either json model or hdf5 model. This function will scan all model by pattern name such (model.json, 
                      weights.h5, weights.hdf5, model.h5, and model.hdf5) UPDATED SOON

                      ACCEPT path of model directory as argument
                      
                      RETURN json_model and hdf5_model which is containing each model path

                      RETURN EXAMPLE : 
                                 * JSON_MODEL : [ ['static/model/BALANCE_model.json', 'static/model/BALANCE_weight.h5'],
                                                  ['static/model/SPLIT_AUGMENTATION_model.json', 'static/model/SPLIT_AUGMENTATION_weight.h5'],
                                                ]

                                 * HDF5_MODEL : ['static/model/BALANCE_model.h5', 'static/model/SPLIT_AUGMENTATION_model.h5']
  """
  json_arch       = []
  json_weight     = []
  hdf5_model      = []
  json_model      = []
  files_in_folder = _getFilesFromFolder(path) # scan all model in path directory

  for data in files_in_folder: # iterate files_in_folder to extract model and weight information

    if "model.json" in data: # get model json by partter name <model.json>
      file_name_and_path = _getFilePathAndName(path, data)
      _appendListElement(json_arch, file_name_and_path)

    elif "weights.h5" in data or "weights.hdf5" in data: # get model weight by partter name <weights.h5 or weights.hdf5>
      file_name_and_path = _getFilePathAndName(path, data)
      _appendListElement(json_weight, file_name_and_path)

    elif "model.h5" in data or "model.hdf5" in data: # get model by partter name <model.h5 or model.hdf5>
      file_name_and_path = _getFilePathAndName(path, data)
      _appendListElement(hdf5_model, file_name_and_path)

  if json_arch and json_weight: # build json model collection (it would return list of json model and realted weight)
    json_model = _getJsonModel(json_arch, json_weight)
  
  return json_model, hdf5_model

def _getDictModel(path):
  """
  _getDictModel() : Provide a collection of model and weight either json or h5 model including model name as keys and model path as values of dictionary
                      This function will help to generate model information for service and application layer. 
                      This function used _buildListModel and _buildDictModel as helper. For detail please see the documentation of each fuction.

                      ACCEPT path of model directory as argument
                      
                      RETURN  dicts, keys and values which is containing model information such path and model name.

                      RETURN EXAMPLE :
                      
                                      * DICTS : {
                                                  'BALANCE_model': 'static/model/BALANCE_model.h5', 
                                                  'IMBALANCE_model': 'static/model/IMBALANCE_model.h5',
                                                  'SPLIT_AUGMENTATION_model': 'static/model/SPLIT_AUGMENTATION_model.h5'
                                                }

                                      * KEYS  : ['BALANCE_model', 'IMBALANCE_model', 'SPLIT_AUGMENTATION_model']

                                      * VALUES: ['static/model/BALANCE_model.h5', 
                                                  'static/model/IMBALANCE_model.h5', 
                                                  'static/model/SPLIT_AUGMENTATION_model.h5']
  """
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
  """
  _loadSelectModel() : This config function used to load selected model. It would help to load model into keras sequential model either json or h5 model.

                      ACCEPT selected model and path of model directory as argument
                      
                      RETURN keras sequential model  <keras.engine.sequential.Sequential object at 0x000002C8C8AB8550>

                      RETURN EXAMPLE :
                      
                                      * LOADED_MODEL :  <keras.engine.sequential.Sequential object at 0x000002C8C8AB8550>
  """
  model_dict, _, _ = _getDictModel(path)

  for data in model_dict:
    
    if data == model:
      model_and_weight  = _getElementByIndex(model_dict, data)

      if type(model_and_weight) == list and model_and_weight: # handle json model and weight
        model_name        = _getElementByIndex(model_and_weight, 0) # get json model name 
        weight_name       = _getElementByIndex(model_and_weight, 1) # get model weight
        json_file         = open(model_name, 'r') # open json model
        loaded_model_json = json_file.read() # read json model
        json_file.close()
        loaded_model      = model_from_json(loaded_model_json) # load json model
        loaded_model.load_weights(weight_name) # load weight

      else: # handle h5 or hdf5 model
        loaded_model      = load_model(model_and_weight)
 
  return loaded_model
    
def _loadCompareModel(list_model, path):
  """
  _loadCompareModel() : This config function used to load selected models. It would help to load all selected model into keras sequential model either json or h5 model.
                        This function will provide a collection of keras sequential model that can be use for service layer.

                      ACCEPT selected list_model and path of model directory as argument
                      
                      RETURN a collection of keras sequential model  [<keras.engine.sequential.Sequential object at 0x000002C8C8AB8550>,
                                                                      <keras.engine.sequential.Sequential object at 0x000002C8C8AB8550>,]

                      RETURN EXAMPLE :
                      
                                      * LIST_OFMODEL :  [<keras.engine.sequential.Sequential object at 0x000002C8C8AB8550>,
                                                        <keras.engine.sequential.Sequential object at 0x000002C8C8AB8550>,]
  """
  model_dict, _, _ = _getDictModel(path)
  list_ofModel = []

  for data in list_model:
    model_and_weight  = _getElementByIndex(model_dict, data)

    if data in model_dict:

      if type(model_and_weight) == list and model_and_weight: # handle json model and weight
        model_name        = _getElementByIndex(model_and_weight, 0) # get json model name 
        weight_name       = _getElementByIndex(model_and_weight, 1) # get model weight
        json_file         = open(model_name, 'r') # open json model
        loaded_model_json = json_file.read() # read json model
        json_file.close()
        loaded_model      = model_from_json(loaded_model_json) # load json model
        loaded_model.load_weights(weight_name) # load weight
        _appendListElement(list_ofModel, loaded_model) # append loaded model with weight into list_OfModel

      else: # handle h5 or hdf5 model
        loaded_model = load_model(model_and_weight) # load h5 or hdf5 model
        _appendListElement(list_ofModel, loaded_model) # append model into list_OfModel
 
  return list_ofModel

def _getJsonModel(models, weights)-> list:
  """
  _getJsonModel() : Provide a collection of json model with each weight. It would be helpfull for build a collection of json model that
                    might be loaded

                      ACCEPT json models and weights of each json model as argument
                      
                      RETURN a collection of json model and each weight

                      RETURN EXAMPLE :
                      
                                      * LIST_OFMODEL :  [<keras.engine.sequential.Sequential object at 0x000002C8C8AB8550>,
                                                        <keras.engine.sequential.Sequential object at 0x000002C8C8AB8550>,]
  """
  sub_value     = []
  json_model    = []
  models.sort(reverse=True) # sort model in dsc term or reverse term
  weights.sort() # sort weight in asc term

  for weight in weights:
    model               = models.pop() # get model name and pop it up
    get_model_and_ext   = _getSplitedStringByIndex(model, "/", -1) # example result: VGG19_model.json
    model_name          = _getSplitedStringByIndex(get_model_and_ext, "_", 0) # example result: VGG19

    if model_name in weight: # check is model_name value is in weight list (find string by pattern)
      _appendListElement(sub_value, model)
      _appendListElement(sub_value, weight)
      _appendListElement(json_model, sub_value)

    sub_value = []

  return json_model

def _rgbImageProcessing(image_file, keras_model):
  """
  _rgbImageProcessing() : Provide a RGB image preprocessing for raw query image based on model input volume information

                      ACCEPT raw image file and keras sequential model as argument
                      
                      RETURN a numpy array of image which is ready to use for prediction

                      RETURN EXAMPLE :
                      
                                      * RESULTIMAGE :   [[[[0.00784314 0.00784314 0.00784314]
                                                            [0.00784314 0.00784314 0.00784314]
                                                            [0.00784314 0.00784314 0.00784314]
                                                            ...
                                                            [0.00392157 0.00392157 0.00392157]
                                                            [0.00392157 0.00392157 0.00392157]
                                                            [0.00392157 0.00392157 0.00392157]]

                                                            ...

                                                            [[0.02352941 0.02352941 0.02352941]
                                                            [0.03529412 0.03529412 0.03529412]
                                                            [0.03137255 0.03137255 0.03137255]
                                                            ...
                                                            [0.02745098 0.02745098 0.02745098]
                                                            [0.02352941 0.02352941 0.02352941]
                                                            [0.01960784 0.01960784 0.01960784]]]]
  """
  readImage           = _openImageFile(image_file) # open image file
  imageNdarray        = _imageToNumpyArray(readImage) # transform image into numpy array
  convertToRGB        = _renderRGBImage(imageNdarray) # change image type from BGR to RGB
  resizeImage ,_ ,_   = _resizeImageByModelInputShape(convertToRGB, keras_model) # resize image based on model input shape
  normalizeImage      = _normalizeImage(resizeImage) # normalize image
  resultImage         = _expandRGBImageDimensions(normalizeImage, 0) # expanding image dimention for prediction

  return resultImage

def _grayImageProcessing(image_file, model):
  """
  _grayImageProcessing() : Provide a Grayscale image preprocessing for raw query image based on model input volume information

                      ACCEPT raw image file and keras sequential model as argument
                      
                      RETURN a numpy array of image which is ready to use for prediction

                      RETURN EXAMPLE :
                      
                                      * RESULTIMAGE :     [[[0.00784314]
                                                            [0.00784314]
                                                            [0.00784314]
                                                            ...
                                                            [0.00392157]
                                                            [0.00392157]
                                                            [0.00392157]]

                                                            ...

                                                            [[0.02352941]
                                                            [0.03529412]
                                                            [0.03137255]
                                                            ...
                                                            [0.02745098]
                                                            [0.02352941]
                                                            [0.01960784]]]
  """
  readImage                    = _openImageFile(image_file) # open image file
  imageNdarray                 = _imageToNumpyArray(readImage) # transform image into numpy array
  convertToRGB                 = _renderRGBImage(imageNdarray) # change image type from BGR to RGB
  convertToGray                = _renderRGBtoGrayImage(convertToRGB) # change image type from RGB into Grayscale
  resizeImage ,_ ,image_size   = _resizeImageByModelInputShape(convertToGray, model) # resize image based on model input shape
  normalizeImage               = _normalizeImage(resizeImage) # normalize image
  resultImage                  = _reshapeGrayImage(normalizeImage, image_size) # expanding image dimention for prediction

  return resultImage