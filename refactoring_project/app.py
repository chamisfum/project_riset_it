"""
Documentation
This part is an application for deploying image classification using Deep Learning algorithm.
Use this part as main service endpoint for managing raw data from frontend into service backend.
It would provide a collection of data such (prediction class, metric performance etc) from service backend into frontend.
"""

# python package
import os
from flask import Flask, request, render_template

# internal package
from src.service import service

"""
Local Config
Comment this part before releasing your application in production
"""
app = Flask(__name__)

"""
Production Config
uncomment and change the static_url_path to into url project path
"""
# app = Flask(__name__, static_url_path='/data_science_product/static')

"""
class_dict is a global variable to define class dictionary 
Keys as <class name> and Value as <class index>
refer to class_indices for flow_from_directory function
"""
class_dict          = {'GLIOMA': 0, 'MENINGIOMA': 1, 'PITUITARY': 2}
# labels is a global variable to get list of class name from class dict
labels  = list(class_dict.keys())
# model_path is a global varible for model path
model_path          = "static/model/"
# query_image_path is a global varible for query image path
query_image_path    = "static/queryImage/"
# query_upload_image is a global varible for uploaded query image path
query_upload_image  = "static/queryUpload/"

"""
IMPORTANT! 
This part would handle after request cache while developing flask api
It would clean and remove all development cache that could be happend
"""
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# ROUTING START
# Render UI template for compare model home page
@app.route("/")
def compare():
    # get list of model that appear in model_path directory
    _, listModel, _             = service.GetDictModel(model_path)
    # get image class and image query file that found in query_image_path directory
    imageClass, _, imageQuery   = service.GetListOfQueryImage(query_image_path)
    # render compare model home page with additional collection of data such model list, image class and image query
    return render_template('/compare.html', listModel=listModel, imageQuery=imageQuery, imageClass=imageClass)

# predict_compare handle a POST method for predict selected image in case comparing several models
@app.route('/pred_comp', methods=['POST'])
def predict_compare():
    # get list of selected models from frontend
    choosenModelList                 = request.form.getlist('select_model') 
    # get selected image
    getImageFile                     = request.form.get('input_image') 
    # get prediction time and result for each selected model
    predictionResult, predictionTime = service.PredictInputRGBImageList(choosenModelList, model_path, getImageFile)
    # render compare model result page with additional collection of data such labels, probs, model names, run_times and image query
    return render_template('/result_compare.html', labels=labels, probs=predictionResult, model=choosenModelList, run_time=predictionTime, img=getImageFile[7:])

# predicts_compare handle a POST method for predict uploaded image in case comparing several models
@app.route('/pred_comps', methods=['POST'])
def predicts_compare():
    # get list of selected models from frontend
    choosenModelList                 = request.form.getlist('select_model')
    # get uploaded an image and save it into query_upload_image folder as temp image
    getImageFile                     = request.files["file"].save(os.path.join(query_upload_image, 'temp.jpg'))
    # get prediction time and result of RGB image for each selected model
    predictionResult, predictionTime = service.PredictInputRGBImageList(choosenModelList, model_path, getImageFile)
    # render compare model result page with additional collection of data such labels, probs, model names, run_times and image query
    return render_template('/result_compare.html', labels=labels, probs=predictionResult, model=choosenModelList, run_time=predictionTime, img='temp.jpg')

# Render UI template for select model home page
@app.route('/select')
def select():
    # get list of model that appear in model_path directory
    _, listModel, _             = service.GetDictModel(model_path)
    # get image class and image query file that found in query_image_path directory
    imageClass, _, imageQuery   = service.GetListOfQueryImage(query_image_path)
    # render select model home page with additional collection of data such model list, image class and image query
    return render_template('/select.html', listModel=listModel, imageQuery=imageQuery, imageClass=imageClass)

# predict_select handle a POST method for predict selected image in case select a model
@app.route('/pred_select', methods=['POST'])
def predict_select():
    # get a selected model from frontend
    choosenModel                     = request.form['select_model']
    # get a selected image
    getImageFile                     = request.form.get('input_image')
    # get prediction time and result of selected model
    predictionResult, predictionTime = service.PredictInputRGBImage(choosenModel, model_path, getImageFile)
    # render compare model result page with additional collection of data such labels, probs, model names, run_times and image query
    return render_template('/result_select.html', labels=labels, probs=predictionResult, model=choosenModel, run_time=predictionTime, img=getImageFile[7:])

@app.route('/pred_selects', methods=['POST'])
def predicts_select():
    # get a selected models from frontend
    choosenModel                     = request.form['select_model']
    # get uploaded an image and save it into query_upload_image folder as temp image
    getImageFile                     = request.files["file"].save(os.path.join(query_upload_image, 'temp.jpg'))
    # get prediction time and result of selected model
    predictionResult, predictionTime = service.PredictInputRGBImage(choosenModel, model_path, getImageFile)
    # render compare model result page with additional collection of data such labels, probs, model names, run_times and image query
    return render_template('/result_select.html', labels=labels, probs=predictionResult, model=choosenModel, run_time=predictionTime, img='temp.jpg')

if __name__ == "__main__": 
        app.run(debug=True, host='0.0.0.0', port=5000)