import os
from flask import Flask, request, render_template
from src import config

app            = Flask(__name__)
class_list     = {'GLIOMA': 0, 'MENINGIOMA': 1, 'PITUITARY': 2}
path           = "static/model/"
queryImagePath = "static/queryImage/"

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def compare():
    _, listModel, _             = config.GetDictModel(path)
    imageClass, _, imageQuery   = config.GetListOfQueryImage(queryImagePath)
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
    _, listModel, _             = config.GetDictModel(path)
    imageClass, _, imageQuery   = config.GetListOfQueryImage(queryImagePath)
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