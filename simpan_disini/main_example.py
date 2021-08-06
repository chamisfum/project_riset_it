import time
import MTCD
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
datahasil = os.listdir('static/result/')

print("Init Flask App")
app = Flask(__name__, static_url_path='/data_science_product/static')
# app = Flask(__name__)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

""" Edit Start """
# Error Handle
@app.route("/")
def index():
    # https://riset.informatika.umm.ac.id/area_of_interest/{aoi_id}
    return redirect('https://riset.informatika.umm.ac.id/area_of_interest/1')

# Compare Model f_nim Brain Tumor Disease
@app.route('/{PRODUCT_ID}/compare')
def f_nim_compare():
    return render_template('/f_nim/compare.html', )

@app.route('/{PRODUCT_ID}/pred_comp', methods=['POST'])
def f_nim_predict_compare(): # Ganti nim dengan NIM anda misal : _2132131312
    respon_model = []
    running_time = []
    chosen_model = request.form.getlist('select_model')
    filename = request.form.get('input_image')
    img = cv2.cvtColor(np.array(np.array(Image.open(filename))), cv2.COLOR_BGR2RGB)
    model_dict = {'Nama Model 1'   :   'static/model/nim/model1.h5', # Isi dengan Nama model dan path lokasi model disimpan (Pastikan menyimpan model dalam folder /static/model/nim/namamodel.h5)
                  'Nama Model 2'   :   'static/model/nim/model2.h5',
                  'Nama json Model 1' :   ['static/model/nim/js/model_js1.json','static/model/nim/js/model_weight_js1.h5'],  # Jika pakai Json dan weight model saat menyimpan model gunakan kode ini
                  'Nama json Model 2'    :   ['static/model/nim/js/model_js2.json','static/model/nim/js/model_weight_js2.h5'] # Beri kode nama "_js" tanpa petik di akhir nama model 
                  }

    for m in chosen_model:
        if "_js" in m: # dari kode nama model _js selanjutnya program akan membaca model format json dengan block kode dalam if
            json_file = open(model_dict[m][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[m][1])
        else: # bila nama model tidak mengandung kode nama _js maka model akan di muat menggunakan load_model()
            model = load_model(model_dict[m])
        
        # preprocessing gambar lakukan sesuai dengan preprocessing yang sama saat proses training
        imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
        
        # mulai prediksi
        start = time.time()
        pred = model.predict(imgs)[0]
        running_time.append(round(time.time()-start,4)) # hitung waktu prediksi
        respon_model.append([round(elem * 100, 2) for elem in pred]) # hitung nilai prediksi
    
    return f_nim_predict_result_compare(respon_model, chosen_model, running_time, filename[7:])

@app.route('/{PRODUCT_ID}/pred_comps', methods=['POST'])
def f_nim_predicts_compare():

    respon_model = []
    running_time = []
    chosen_model = request.form.getlist('select_model')
    file = request.files["file"]
    file.save(os.path.join('static', 'temp.jpg'))
    img = cv2.cvtColor(np.array(np.array(Image.open(file))), cv2.COLOR_BGR2RGB)
    model_dict = {'Nama Model 1'   :   'static/model/nim/model1.h5', # Isi dengan Nama model dan path lokasi model disimpan (Pastikan menyimpan model dalam folder /static/model/nim/namamodel.h5)
                  'Nama Model 2'   :   'static/model/nim/model2.h5',
                  'Nama json Model 1' :   ['static/model/nim/js/model_js1.json','static/model/nim/js/model_weight_js1.h5'],  # Jika pakai Json dan weight model saat menyimpan model gunakan kode ini
                  'Nama json Model 2'    :   ['static/model/nim/js/model_js2.json','static/model/nim/js/model_weight_js2.h5'] # Beri kode nama "_js" tanpa petik di akhir nama model 
                  }

    for m in chosen_model:
        if "_js" in m: # dari kode nama model _js selanjutnya program akan membaca model format json dengan block kode dalam if
            json_file = open(model_dict[m][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[m][1])
        else: # bila nama model tidak mengandung kode nama _js maka model akan di muat menggunakan load_model()
            model = load_model(model_dict[m])
        
        # preprocessing gambar lakukan sesuai dengan preprocessing yang sama saat proses training
        imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
        
        # mulai prediksi
        start = time.time()
        pred = model.predict(imgs)[0]
        running_time.append(round(time.time()-start,4)) # hitung waktu prediksi
        respon_model.append([round(elem * 100, 2) for elem in pred]) # hitung nilai prediksi

    return f_nim_predict_result_compare(respon_model, chosen_model, running_time, 'temp.jpg')

def f_nim_predict_result_compare(probs, mdl, run_time, img):

    class_list = {'Nama Kelas 1': 0, 'Nama Kelas 2': 1} # isi dengan nama kelas 1 sampai ke n sesuai dengan urutan kelas data pada classification report key di isi dengan nama kelas dan value di isi dengan urutan kelas dimulai dari 0
    idx_pred = [i.index(max(i)) for i in probs]
    labels = list(class_list.keys())
    return render_template('/nim/result_compare.html', labels=labels, 
                            probs=probs, mdl=mdl, run_time=run_time, pred=idx_pred, img=img)

# Select Model f_nim Brain Tumor Disease
@app.route('/{PRODUCT_ID}/select')
def f_nim_select():
    return render_template('/nim/select.html', )

@app.route('/{PRODUCT_ID}/pred_select', methods=['POST'])
def f_nim_predict_select():
    
    chosen_model = request.form['select_model']
    model_dict = {'Nama Model 1'   :   'static/model/nim/model1.h5', # Isi dengan Nama model dan path lokasi model disimpan (Pastikan menyimpan model dalam folder /static/model/nim/namamodel.h5)
                  'Nama Model 2'   :   'static/model/nim/model2.h5',
                  'Nama json Model 1' :   ['static/model/nim/js/model_js1.json','static/model/nim/js/model_weight_js1.h5'],  # Jika pakai Json dan weight model saat menyimpan model gunakan kode ini
                  'Nama json Model 2'    :   ['static/model/nim/js/model_js2.json','static/model/nim/js/model_weight_js2.h5'] # Beri kode nama "_js" tanpa petik di akhir nama model 
                  }
    if chosen_model in model_dict:
        if "_js" in m: # dari kode nama model _js selanjutnya program akan membaca model format json dengan block kode dalam if
            json_file = open(model_dict[m][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[m][1])
        else: # bila nama model tidak mengandung kode nama _js maka model akan di muat menggunakan load_model()
            model = load_model(model_dict[m])
    else:
        model = load_model(model_dict[0]) # load default model
    
    filename = request.form.get('input_image')
    
    # preprocessing gambar lakukan sesuai dengan preprocessing yang sama saat proses training
    img = cv2.cvtColor(np.array(Image.open(filename)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    
    # mulai prediki
    start = time.time()
    pred = model.predict(img)[0]
    runtimes = round(time.time()-start,4) # hitung lama prediksi

    respon_model = [round(elem * 100, 2) for elem in pred] # hitung nilai prediksi

    return f_nim_predict_result_select(chosen_model, runtimes, respon_model, filename[7:])

@app.route('/{PRODUCT_ID}/pred_selects', methods=['POST'])
def f_nim_predicts_select():

    chosen_model = request.form['select_model']
    model_dict = {'Nama Model 1'   :   'static/model/nim/model1.h5', # Isi dengan Nama model dan path lokasi model disimpan (Pastikan menyimpan model dalam folder /static/model/nim/namamodel.h5)
                  'Nama Model 2'   :   'static/model/nim/model2.h5',
                  'Nama json Model 1' :   ['static/model/nim/js/model_js1.json','static/model/nim/js/model_weight_js1.h5'],  # Jika pakai Json dan weight model saat menyimpan model gunakan kode ini
                  'Nama json Model 2'    :   ['static/model/nim/js/model_js2.json','static/model/nim/js/model_weight_js2.h5'] # Beri kode nama "_js" tanpa petik di akhir nama model 
                  } 

    if chosen_model in model_dict:
        if "_js" in chosen_model: # dari kode nama model _js selanjutnya program akan membaca model format json dengan block kode dalam if
            json_file = open(model_dict[chosen_model][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[chosen_model][1])
        else: # bila nama model tidak mengandung kode nama _js maka model akan di muat menggunakan load_model()
            model = load_model(model_dict[chosen_model])
    else:
        model = load_model(model_dict[0]) # load default model
    
    file = request.files["file"]
    file.save(os.path.join('static', 'temp.jpg'))

    # preprocessing gambar lakukan sesuai dengan preprocessing yang sama saat proses training
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    
    # mulai prediki
    start = time.time()
    pred = model.predict(img)[0]
    runtimes = round(time.time()-start,4) # hitung lama prediksi

    respon_model = [round(elem * 100, 2) for elem in pred] # hitung nilai prediksi

    return f_nim_predict_result_select(chosen_model, runtimes, respon_model, 'temp.jpg')

def f_nim_predict_result_select(model, run_time, probs, img):
    class_list = {'Nama Kelas 1': 0, 'Nama Kelas 2': 1} # isi dengan nama kelas 1 sampai ke n sesuai dengan urutan kelas data pada classification report key di isi dengan nama kelas dan value di isi dengan urutan kelas dimulai dari 0
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/nim/result_select.html', labels=labels, 
                            probs=probs, model=model, pred=idx_pred, 
                            run_time=run_time, img=img)

""" Edit End """

if __name__ == "__main__": 
        # app.run(debug=True, host='0.0.0.0', port=2000)
        app.run(debug=False, host='0.0.0.0', port=2000, ssl_context = ('/home/admin/conf/web/ssl.riset.informatika.umm.ac.id.crt','/home/admin/conf/web/ssl.riset.informatika.umm.ac.id.key'))
