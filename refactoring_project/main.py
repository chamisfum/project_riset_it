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
from src import config

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
class_list is a global variable to define class dictionary 
Keys as <class name> and Value as <class index>
refer to class_indices for flow_from_directory function
instead using dictionary u can use an ordred class list
"""
# class_list  = ['GLIOMA', 'MENINGIOMA', 'PITUITARY']
class_dict          = {'GLIOMA': 0, 'MENINGIOMA': 1, 'PITUITARY': 2}
# model_path is a global varible for model path
model_path          = "static/model/"
# query_image_path is a global varible for query image path
query_image_path    = "static/queryImage/"

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
# Render UI template for compare home page
@app.route("/")
def compare():
    _, listModel, _             = config.GetDictModel(model_path)
    imageClass, _, imageQuery   = config.GetListOfQueryImage(query_image_path)
    return render_template('/compare.html', listModel=listModel, imageQuery=imageQuery, imageClass=imageClass)

@app.route('/pred_comp', methods=['POST'])
def predict_compare():
    choosenModelList                 = request.form.getlist('select_model')
    getImageFile                     = request.form.get('input_image') 
    loadedModelList                  = config.LoadCompareModel(choosenModelList, path)
    predictionResult, predictionTime = config.PredictInputRGBImageList(loadedModelList, getImageFile)
    return predict_result_compare(choosenModelList, predictionTime, predictionResult, getImageFile[7:])

@app.route('/pred_comps', methods=['POST'])
def predicts_compare():
    choosenModelList                 = request.form.getlist('select_model')
    getImageFile                     = request.files["file"].save(os.path.join('static/query', 'temp.jpg'))
    loadedModelList                  = config.LoadCompareModel(choosenModelList, path)
    predictionResult, predictionTime = config.PredictInputRGBImageList(loadedModelList, getImageFile)
    return predict_result_compare(choosenModelList, predictionTime, predictionResult, 'temp.jpg')

def predict_result_compare(model, runTime, result, image):
    labels      = list(class_list.keys())
    return render_template('/result_compare.html', labels=labels, probs=result, 
                            mdl=model, run_time=runTime, pred=result, img=image)

@app.route('/select')
def select():
    _, listModel, _             = config.GetDictModel(model_path)
    imageClass, _, imageQuery   = config.GetListOfQueryImage(query_image_path)
    return render_template('/select.html', listModel=listModel, imageQuery=imageQuery, imageClass=imageClass)

@app.route('/pred_select', methods=['POST'])
def predict_select():
    choosenModel                     = request.form['select_model']
    getImageFile                     = request.form.get('input_image')
    loadedModel                      = config.LoadSelectModel(choosenModel, path)
    queryImage                       = config.ImageProcessingRGB(getImageFile, loadedModel)
    predictionResult, predictionTime = config.PredictInputImage(loadedModel, queryImage)
    return predict_result_select(choosenModel, predictionTime, predictionResult, getImageFile[7:])

@app.route('/pred_selects', methods=['POST'])
def predicts_select():
    choosenModel                     = request.form['select_model']
    getImageFile                     = request.files["file"].save(os.path.join('static/query', 'temp.jpg'))
    loadedModel                      = config.LoadSelectModel(choosenModel, path)
    queryImage                       = config.ImageProcessingRGB(getImageFile, loadedModel)
    predictionResult, predictionTime = config.PredictInputImage(loadedModel, queryImage)
    return predict_result_select(choosenModel, predictionTime, predictionResult, 'temp.jpg')

def predict_result_select(model, runTime, result, image):
    idxPredictionList = result.index(max(result))
    labels            = list(class_list.keys())
    return render_template('/result_select.html', labels=labels, probs=result, model=model, 
                            pred=idxPredictionList, run_time=runTime, img=image)

if __name__ == "__main__": 
        app.run(debug=True, host='0.0.0.0', port=5000)