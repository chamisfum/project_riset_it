"""

Documentation

This part is an application for deploying image classification using Deep Learning algorithm. 
Use this part as main service endpoint for managing raw data from frontend into service backend. 
It would provide a collection of data such (prediction class, metric performance etc) 
from service backend into frontend.

"""

# python package
# import requests
# assert requests.get('https://github.com/nvbn/import_from_github_com').status_code == 200
from flask import Flask, request, render_template

# internal package
from src.service import service

# initialize global function alias
GetFilePathAndName              = service._getFilePathWithName
ModelDictionary                 = service._getDictModel
QueryImageList                  = service.GetListOfQueryImage
PredictRGBImageList             = service.PredictInputRGBImageList
PredicRGBImage                  = service.PredictInputRGBImage

"""
GLOBAL CONSTANT VARIABLE!
    * class_dict is a global dictionary variable. 
        Possible value : Keys as <class name> and Value as <class index> 
        refer to class_indices from flow_from_directory function
    * labels is a global list variable of class name from class dictionary
    * model_path is a global varible for model path
    * query_image_path is a global varible for query image path
    * query_upload_image is a global varible for uploaded query image path
"""
CLASS_DICT          = {'GLIOMA': 0, 'MENINGIOMA': 1, 'PITUITARY': 2}
LABELS              = list(CLASS_DICT.keys())
MODEL_PATH          = "static/model/"
QUERY_IMAGE_PATH    = "static/queryImage/"
QUERY_UPLOAD_IMAGE  = "static/queryUpload/"

# IMPORTANT!
# please change this part into your product detail and configuration
PARENT_LOCATION     = "data_science_product" # refers to parent of project web service configuration
TOPIC_NAME          = "Brain Tumor Disease" # represent 
AREA_OF_INTEREST_ID = "1"
TOPIC_ID            = "1"
PRODUCT_ID          = "1"

"""
LOCAL CONFIG!
    Comment this part before releasing your application in production
"""
app = Flask(__name__)

"""
PRODUCTION CONFIG!
    uncomment and change the static_url_path to into url project path
"""
# app = Flask(__name__, static_url_path=PARENT_LOCATION+'static')

@app.after_request
def add_header(r):
    """
        IMPORTANT! 
        This part would handle after request cache while developing flask api
        It would clean and remove all development cache that could be happend
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

#   ROUTING START!
@app.route("/")
def compare():
    """
        Render UI template for compare model home page
        This part will provide a collection of model name and path, also class 
            name of each example of query images and image full path.
    """
    _, listModel, _             = ModelDictionary(MODEL_PATH)
    imageClass, _, imageQuery   = QueryImageList(QUERY_IMAGE_PATH)
    return render_template('/compare.html', listModel = listModel, parent_location = PARENT_LOCATION,
                            topic_name = TOPIC_NAME, aoi_id = AREA_OF_INTEREST_ID, topic_id = TOPIC_ID, 
                            product_id = PRODUCT_ID, imageQuery = imageQuery, imageClass=imageClass)

@app.route('/pred_comp', methods=['POST'])
def predict_compare():
    """
    PREDICT_COMPARE : handle POST prediction of selected image in case comparing several models
                    * choosenModelList get list of selected models from frontend
                    * getImageFile get selected image from frontend
                    * PredictRGBImageList() provide a collection of prediction time and result
                    * predictionTime hold prediction time and
                    * predictionResult hold prediction result for each selected model
                    * render compare model result page with a collection of data such
                    * (labels, probs, model names, run_times and image query)
    """
    choosenModelList                 = request.form.getlist('select_model') 
    getImageFile                     = request.form.get('input_image') 
    predictionResult, predictionTime = PredictRGBImageList(choosenModelList, MODEL_PATH, getImageFile)
    return render_template('/result_compare.html', labels = LABELS, probs = predictionResult, parent_location = PARENT_LOCATION,
                            topic_name = TOPIC_NAME, aoi_id = AREA_OF_INTEREST_ID, topic_id = TOPIC_ID, 
                            product_id = PRODUCT_ID, model = choosenModelList, run_time = predictionTime, img = getImageFile[7:])

@app.route('/pred_comps', methods=['POST'])
def predicts_compare():
    """
    PREDICTS_COMPARE : handle POST prediction of uploaded image in case comparing several models
                     * choosenModelList get list of selected models from frontend
                     * getImageFile get uploaded image from frontend and save as temp image 
                     * PredictRGBImageList() provide a collection of prediction time and result
                     * predictionTime hold prediction time and 
                     * predictionResult hold prediction result for each selected model
                     * render compare model result page with a collection of data such
                     * (labels, probs, model names, run_times and image query)
    """
    choosenModelList                 = request.form.getlist('select_model')
    getImageFile                     = request.files["file"]
    relocationImageFile              = GetFilePathAndName(QUERY_UPLOAD_IMAGE, 'temp.jpg')
    getImageFile.save(relocationImageFile)
    predictionResult, predictionTime = PredictRGBImageList(choosenModelList, MODEL_PATH, getImageFile)
    return render_template('/result_compare.html', labels = LABELS, probs = predictionResult, parent_location = PARENT_LOCATION,
                            topic_name = TOPIC_NAME, aoi_id = AREA_OF_INTEREST_ID, topic_id = TOPIC_ID, 
                            product_id = PRODUCT_ID, model = choosenModelList, run_time = predictionTime, img = relocationImageFile)

@app.route('/select')
def select():
    """
        Render UI template for select model home page
        This part will provide a collection of model name and path, also class name of each 
        example of query images and image full path.
    """
    _, listModel, _             = ModelDictionary(MODEL_PATH)
    imageClass, _, imageQuery   = QueryImageList(QUERY_IMAGE_PATH)
    return render_template('/select.html', listModel = listModel, parent_location = PARENT_LOCATION,
                            topic_name = TOPIC_NAME, aoi_id = AREA_OF_INTEREST_ID, topic_id = TOPIC_ID, 
                            product_id = PRODUCT_ID, imageQuery = imageQuery, imageClass = imageClass)

@app.route('/pred_select', methods=['POST'])
def predict_select():
    """
    PREDICT_SELECT  : handle POST prediction of selected image
                    * choosenModelList get selected models from frontend
                    * getImageFile get an selected image from frontend
                    * PredicRGBImage() give prediction time and result
                    * predictionTime hold prediction time and 
                    * predictionResult hold prediction result for each selected model
                    * render compare model result page with a collection of data such
                    * (labels, probs, model names, run_times and image query)
    """
    choosenModel                     = request.form['select_model']
    getImageFile                     = request.form.get('input_image')
    predictionResult, predictionTime = PredicRGBImage(choosenModel, MODEL_PATH, getImageFile)
    return render_template('/result_select.html', labels = LABELS, probs = predictionResult, parent_location = PARENT_LOCATION,
                            topic_name = TOPIC_NAME, aoi_id = AREA_OF_INTEREST_ID, topic_id = TOPIC_ID, 
                            product_id = PRODUCT_ID, model = choosenModel, run_time = predictionTime, img = getImageFile[7:])

@app.route('/pred_selects', methods=['POST'])
def predicts_select():
    """
    PREDICTS_SELECT : handle POST prediction of uploaded image
                    * choosenModelList get selected models from frontend
                    * getImageFile get an uploaded image from frontend and save as temp image 
                    * PredicRGBImage() give prediction time and result
                    * predictionTime hold prediction time and
                    * predictionResult hold prediction result for each selected model
                    * render compare model result page with a collection of data such 
                    * (labels, probs, model names, run_times and image query)
    """
    choosenModel                     = request.form['select_model']
    getImageFile                     = request.files["file"]
    relocationImageFile              = GetFilePathAndName(QUERY_UPLOAD_IMAGE, 'temp.jpg')
    getImageFile.save(relocationImageFile)
    predictionResult, predictionTime = PredicRGBImage(choosenModel, MODEL_PATH, getImageFile)
    return render_template('/result_select.html', labels = LABELS, probs = predictionResult, parent_location = PARENT_LOCATION,
                            topic_name = TOPIC_NAME, aoi_id = AREA_OF_INTEREST_ID, topic_id = TOPIC_ID, 
                            product_id = PRODUCT_ID, model = choosenModel, run_time = predictionTime, img = relocationImageFile)

if __name__ == "__main__": 
    # LOCAL DEVELOPMENT CONFIG
    app.run(debug=True, host='0.0.0.0', port=5000)
"""
    PRODUCTION CONFIG
    app.run(debug=False, host='0.0.0.0', port=2000, 
    # handle ssl cert an keys
            ssl_context = ('/home/admin/conf/web/ssl.riset.informatika.umm.ac.id.crt',
                          '/home/admin/conf/web/ssl.riset.informatika.umm.ac.id.key'))
"""
