# Riset Informatika Universitas Muhammadiyah Malang / Informatics Research Center

<p align="center">
  <img src="main-logo.png" alt="Logo Riset Informatika"/>
</p>

Riset Informatika Universitas Muhammadiyah Malang adalah sebuah platform ...

## Table of Contents

- [Manual Book](#manual-book)
- [Installation](#installation)
    - [Create Virtual Environment with virtualenv](#create-env-virtualenv)
        - [Activate virtualenv with Windows](#activate-virtualenv-windows)
        - [Activate virtualenv with Linux or MacOS](#activate-virtualenv-linux-macos)
    - [Create Virtual Environment with Anaconda](#create-env-conda)
    - [Install Library](#install-library)
- [To Do List](#todo-list)
- [Run Flask on Localhost](#run-flask-local)

## Manual Book

- [Researcher Manual Book](https://drive.google.com/file/d/14t3G5CNSveTThrlBv7JyKfhUKF16J9w3/view)
- [Public Guide Book](https://drive.google.com/file/d/1UIFmSPu-YoRmr2JkFj5NXKB-iFYZRC-W/view)

## Installation

Install terlebih dahulu `requirements.txt` di environment masing - masing atau membuat virtual environment terlebih dahulu dengan `virtualenv` atau `Anaconda`.

### Create Virtual Environment with virtualenv

```sh
pip install virtualenv
virtualenv project-riset-it
```

### Activate virtualenv with Windows
```sh
\pathto\project-riset-it\Scripts\activate.bat
```

### Activate virtualenv with Linux or MacOS
```sh
source project-riset-it/bin/activate
```

### Create Virtual Environment with Anaconda
```sh
conda create --name project-riset-it
conda activate project-riset-it
```

### Install Library

```py
pip install -r requirements.txt
```

## To Do List

Beberapa Code yang harus diganti sesuai dengan kebutuhan masing - masing. Tetapi ada beberapa function atau method yang tidak harus sesuai dengan repository ini, sesuaikan dengan use case masing - masing.

- [Python](#python)
    - [Ganti NIM](#ganti-nim)
    - [Ganti Model](#ganti-model)
    - [Ganti Preprocessing](#ganti-preprocessing)
    - [Ganti Label](#ganti-label)
    - [Menambahkan Function Sendiri](#tambah-function) (Optional)
- [HTML](#html)
    - [NIM](#nim)
    - [General](#general)
    - [Compare](#compare)
    - [Select](#select)

### Upload Your Model

Ada dua jenis format model yang bisa dipakai disini, yaitu `.json` dan `.h5`. Masing - masing model di upload di folder `simpan_disini/static/model/nim/` untuk model dengan format `.h5`, jika model menggunakan `.json` bisa upload di folder `simpan_disini/static/model/nim/js/`.

## Python
----------

### Ganti NIM

Mengganti semua function Python di dalam Flask dan folder path yang mengandung `_nim` dan `nim` dengan NIM masing - masing.

***simpan_disini/main_example.py***

```py
@app.route('/{PRODUCT_ID}/compare')
def f_nim_compare():
    return render_template('/f_nim/compare.html', )

@app.route('/{PRODUCT_ID}/pred_comp', methods=['POST'])
def f_nim_predict_compare(): # Ganti nim dengan NIM anda misal : _2132131312
    ...
    return f_nim_predict_result_compare(respon_model, chosen_model, running_time, filename[7:])

@app.route('/{PRODUCT_ID}/select')
def f_nim_select():
    return render_template('/nim/select.html', )

@app.route('/{PRODUCT_ID}/pred_select', methods=['POST'])
def f_nim_predict_select():
    ...
    return f_nim_predict_result_select(chosen_model, runtimes, respon_model, filename[7:])
```

Example:

```py
@app.route('/{PRODUCT_ID}/compare')
def f_201710370311000_compare():
    return render_template('/f_201710370311000/compare.html', )

@app.route('/{PRODUCT_ID}/pred_comp', methods=['POST'])
def f_201710370311000_predict_compare(): # Ganti nim dengan NIM anda misal : _2132131312
    ...
    return f_201710370311000_predict_result_compare(respon_model, chosen_model, running_time, filename[7:])

@app.route('/{PRODUCT_ID}/select')
def f_201710370311000_select():
    return render_template('/nim/select.html', )

@app.route('/{PRODUCT_ID}/pred_select', methods=['POST'])
def f_201710370311000_predict_select():
    ...
    return f_201710370311000_predict_result_select(chosen_model, runtimes, respon_model, filename[7:])
```

### Ganti Model

Mengganti key dan value pada dictionary model sesuai dengan kebutuhan masing - masing. Jika model dengan format `.json` juga harus menambahkan **weights** dari model dengan format `.h5`. Ada dua jenis **function** yang disediakan pada repository ini, yaitu **Compare Model** dan **Select Model**. Jika ingin melakukan compare, edit key dan value pada function `_compare()` dan jika model yang digunakan hanya satu atau ingin menggunakan salah satu model yang diinginkan, gunakan function `_select()`.

***simpan_disini/main_example.py***

```py
@app.route('/{PRODUCT_ID}/pred_select', methods=['POST'])
def f_nim_predict_select():
    
    chosen_model = request.form['select_model']
    model_dict = {'Nama Model 1'   :   'static/model/nim/model1.h5', # Isi dengan Nama model dan path lokasi model disimpan (Pastikan menyimpan model dalam folder /static/model/nim/namamodel.h5)
                  'Nama Model 2'   :   'static/model/nim/model2.h5',
                  'Nama json Model 1' :   ['static/model/nim/js/model_js1.json','static/model/nim/js/model_weight_js1.h5'],  # Jika pakai Json dan weight model saat menyimpan model gunakan kode ini
                  'Nama json Model 2'    :   ['static/model/nim/js/model_js2.json','static/model/nim/js/model_weight_js2.h5'] # Beri kode nama "_js" tanpa petik di akhir nama model 
                  }
    ...
```

Example:

```py
@app.route('/{PRODUCT_ID}/pred_select', methods=['POST'])
def f_nim_predict_select():
    
    chosen_model = request.form['select_model']
    model_dict = {'ResNet50'   :   'static/model/nim/resnet50.h5', # Isi dengan Nama model dan path lokasi model disimpan (Pastikan menyimpan model dalam folder /static/model/nim/namamodel.h5)
                  'BaseCNN'   :   'static/model/nim/BaseCNN.h5',
                  'ModelKeren' :   ['static/model/nim/js/ModelKeren_js.json','static/model/nim/js/ModelKeren_weight_js1.h5'],  # Jika pakai Json dan weight model saat menyimpan model gunakan kode ini
                  'ModelSuperKeren'    :   ['static/model/nim/js/ModelSuperKeren_js.json','static/model/nim/js/ModelSuperKeren_weight_js.h5'] # Beri kode nama "_js" tanpa petik di akhir nama model 
                  }
    ...
```

### Ganti Preprocessing

Pada proses ini, sesuaikan dengan use case masing - masing dan pada saat proses training model.

***simpan_disini/main_example.py***

```py
@app.route('/{PRODUCT_ID}/pred_select', methods=['POST'])
def f_nim_predict_select():
    ...
    imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
```

Example:

```py
@app.route('/{PRODUCT_ID}/pred_select', methods=['POST'])
def f_nim_predict_select():
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return image
```

### Ganti Label

Mengganti label kelas sesuai dengan use case masing - masing.

***simpan_disini/main_example.py***

```py
def f_nim_predict_result_select(model, run_time, probs, img):
    class_list = {'Nama Kelas 1': 0, 'Nama Kelas 2': 1} # isi dengan nama kelas 1 sampai ke n sesuai dengan urutan kelas data pada classification report key di isi dengan nama kelas dan value di isi dengan urutan kelas dimulai dari 0
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/nim/result_select.html', labels=labels, 
                            probs=probs, model=model, pred=idx_pred, 
                            run_time=run_time, img=img)
```

Example:

```py
def f_nim_predict_result_select(model, run_time, probs, img):
    class_list = {'Otsu Thresholding': 0, 'Niblack Thresholding': 1, 'Sauvola Thresholding': 2} # isi dengan nama kelas 1 sampai ke n sesuai dengan urutan kelas data pada classification report key di isi dengan nama kelas dan value di isi dengan urutan kelas dimulai dari 0
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/nim/result_select.html', labels=labels, 
                            probs=probs, model=model, pred=idx_pred, 
                            run_time=run_time, img=img)
```

### Menambahkan Function Sendiri

Proses ini termasuk optional, karena pada repository ini lebih melakukan demo pada use case Image Classification. Jika use case yang dikerjakan seperti Generative Adversarial Network (GAN), Image Segmentation, dll bisa membuat function sendiri.

Example:

### Create Image Segmentation Function

***simpan_disini/segmentation.py***

```py
from skimage.filters import threshold_otsu

def otsu_thresh(file, img_width, img_height):
    '''
    Konversi image menjadi binary menggunakan metode Otsu Thresholding
    Parameters
    ----------
    file: string
        Path ke file image yang kita inginkan
    
    img_width: int
        Size untuk width gambar kita
    
    img_height: int
        Size untuk height gambar kita
    Returns
    -------
    binary_global: Boolean
        masih belum tahu hehe
    '''

    image = cv2.imread(file, 0)
    image = cv2.resize(image, (img_width, img_height))

    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
```

***simpan_disini/main_example.py***

```py
from segmentation import otsu_thresh
...
# Apply your own function in here
```

## HTML
--------

Ada beberapa yang harus diganti, [NIM](#nim), [General](#general), [Compare](#compare), [Select](#select). Base folder ini di `simpan_disini/templates/nim/`.

### NIM

Pada bagian ini, harap mengganti `nim` pada folder `simpan_disini/templates/nim` sesuai dengan NIM masing - masing.

### General

Pada proses ini mengganti bagian dibawah ini pada masing - masing file seperti `compare.html`, `result_compare.html`, `result_select.html`, `select.html`.

```html
        <div class="container">
            <div class="content-box">
                    <div class="title">Research Product Details</div>
                    <ul class="bread-crumb">
                        <li><a href="https://riset.informatika.umm.ac.id">Home</a></li>
                        <li><a href="https://riset.informatika.umm.ac.id/area_of_interest/1">Data Science</a></li>
                        <li><a href="https://riset.informatika.umm.ac.id/area_of_interest/topics/1">Research Topics</a></li>
                    <!-- EDIT START-->
                        <li><a href="https://riset.informatika.umm.ac.id/area_of_interest/topics/{TOPIC_ID}/all">{NAMA TOPIC}</a></li>
                        <li><a href="{{parent_location}}{PRODUCT_ID}/select">Predict</a></li>
                        <li><a href="{{parent_location}}{PRODUCT_ID}/compare">Compare</a></li>
    
                        <!-- JIKA HANYA ADA 1 MODEL GUNAKAN KODE DIBAWAH INI -->
                        
                        <li><a href="https://riset.informatika.umm.ac.id/area_of_interest/topics/{TOPIC_ID}/all">{NAMA TOPIC}</a></li>
                        <li>Product</li>
                        
                    <!-- EDIT END -->
```

Example:

```html
        <div class="container">
            <div class="content-box">
                    <div class="title">Research Product Details</div>
                    <ul class="bread-crumb">
                        <li><a href="https://riset.informatika.umm.ac.id">Home</a></li>
                        <li><a href="https://riset.informatika.umm.ac.id/area_of_interest/1">Data Science</a></li>
                        <li><a href="https://riset.informatika.umm.ac.id/area_of_interest/topics/1">Research Topics</a></li>
                    <!-- EDIT START-->
                        <li><a href="https://riset.informatika.umm.ac.id/area_of_interest/topics/11/all">Klasifikasi Tumor Otak Menggunakan SVM</a></li>
                        <li><a href="{{parent_location}}{PRODUCT_ID}/select">Predict</a></li>
                        <li><a href="{{parent_location}}{PRODUCT_ID}/compare">Compare</a></li>
    
                        <!-- JIKA HANYA ADA 1 MODEL GUNAKAN KODE DIBAWAH INI -->
                        
                        <li><a href="https://riset.informatika.umm.ac.id/area_of_interest/topics/11/all">Klasifikasi Tumor Otak Menggunakan SVM</a></li>
                        <li>Product</li>
                        
                    <!-- EDIT END -->
            ...
```

### Compare

Pada proses ini edit file `compare.html`.

***simpan_disini/templates/nim/compare.html***

```html
    <!-- form-section -->
    <section class="contact-section" style="padding-top:50px;" id="predict1">
        <div class="container">
            <div class="title-box centred" >
                <div class="sec-title"><a class="sec-title" href="#predict1">Upload an Image</a> / <a class="sec-title" href="#predict">Select an Image</a><br> to Classify</div><br>
            </div>
        <!-- EDIT START-->
            <form class="login100-form validate-form" action="{{parent_location}}11/pred_comps" method=post enctype=multipart/form-data>
        <!-- EDIT END -->
                <div class="row" style=" width: 100%; margin: auto; padding: 15px;" >
                    <div class="column" >
                        <div class="form-check form-check-inline">
                            <p style="font-weight: bold; color: black;">Compare Model : </p>
                        </div>   
                    </div>
                    <div class="column" >
                        <div class="form-check form-check-inline"> 
                            <div class="checkboxes">
                            <!-- EDIT START-->
                                <input type="hidden" name="select_model" id="chose_model" value="model1">
                                <label style="padding-left:2em"><input name="select_model" type="checkbox" id="chose_model" value='model1'   checked disabled/> <span>Nama Model 1 </span></label>
                                <label style="padding-left:2em"><input name="select_model" type="checkbox" id="chose_model" value='model2'     /> <span>Nama Model 2 </span></label>
                                <label style="padding-left:2em"><input name="select_model" type="checkbox" id="chose_model" value='model_js1'  /> <span>Nama json Model 1 </span></label>
                                <label style="padding-left:2em"><input name="select_model" type="checkbox" id="chose_model" value='model_js2'    /> <span>Nama json Model 2 </span></label>
                            <!-- EDIT END -->
                            </div>
                        </div>
                    </div>
                </div>
            ...
            <!-- EDIT START-->
                <div class="row">
                        <div class="column" style="background-color:rgb(235, 235, 235);">
                            <p style="text-align: center;">{LABEL DATA}</p>
                            <input type="hidden" name="input_image" id="count" value="static/main/images/predict/{NIM}/gambar1.jpg" />
                            <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar1.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar1.jpg')}}"/>
                        </div>
                        <div class="column" style="background-color:rgb(235, 235, 235);">
                            <p style="text-align: center;">{LABEL DATA}</p>
                            <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar2.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar2.jpg')}}"/>
                        </div>
                        <div class="column" style="background-color:rgb(235, 235, 235);">
                            <p style="text-align: center;">{LABEL DATA}</p>
                            <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar3.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar3.jpg')}}"/>
                        </div>
                        <div class="column" style="background-color:rgb(235, 235, 235);">
                            <p style="text-align: center;">{LABEL DATA}</p>
                            <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar4.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar4.jpg')}}"/>
                        </div>
                        <div class="column" style="background-color:rgb(235, 235, 235);">
                            <p style="text-align: center;">{LABEL DATA}</p>
                            <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar5.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar5.jpg')}}"/>
                        </div>
                </div>  
            ...

```

Example:

```html
    <!-- form-section -->
    <section class="contact-section" style="padding-top:50px;" id="predict1">
        <div class="container">
            <div class="title-box centred" >
                <div class="sec-title"><a class="sec-title" href="#predict1">Upload an Image</a> / <a class="sec-title" href="#predict">Select an Image</a><br> to Classify</div><br>
            </div>
        <!-- EDIT START-->
            <form class="login100-form validate-form" action="{{parent_location}}11/pred_comps" method=post enctype=multipart/form-data>
        <!-- EDIT END -->
                <div class="row" style=" width: 100%; margin: auto; padding: 15px;" >
                    <div class="column" >
                        <div class="form-check form-check-inline">
                            <p style="font-weight: bold; color: black;">Compare Model : </p>
                        </div>   
                    </div>
                    <div class="column" >
                        <div class="form-check form-check-inline"> 
                            <div class="checkboxes">
                            <!-- EDIT START-->
                                <input type="hidden" name="select_model" id="chose_model" value="model1">
                                <label style="padding-left:2em"><input name="select_model" type="checkbox" id="chose_model" value='model1'   checked disabled/> <span>ResNet 50 </span></label>
                                <label style="padding-left:2em"><input name="select_model" type="checkbox" id="chose_model" value='model2'     /> <span>BaseCNN </span></label>
                                <label style="padding-left:2em"><input name="select_model" type="checkbox" id="chose_model" value='model_js1'  /> <span>ModelKeren JSON</span></label>
                                <label style="padding-left:2em"><input name="select_model" type="checkbox" id="chose_model" value='model_js2'    /> <span>ModelSuperKeren JSON</span></label>
                            <!-- EDIT END -->
                            </div>
                        </div>
                    </div>
                </div>
                ...

                    <!-- EDIT START-->
                <div class="row">
                        <div class="column" style="background-color:rgb(235, 235, 235);">
                            <p style="text-align: center;">Otsu Thresholding</p>
                            <input type="hidden" name="input_image" id="count" value="static/main/images/predict/201710370311000/gambar1.jpg" />
                            <input type="image" name="submit" onclick="change('static/main/images/predict/201710370311000/gambar1.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/201710370311000/gambar1.jpg')}}"/>
                        </div>
                        <div class="column" style="background-color:rgb(235, 235, 235);">
                            <p style="text-align: center;">Niblack Thresholding</p>
                            <input type="image" name="submit" onclick="change('static/main/images/predict/201710370311000/gambar2.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/201710370311000/gambar2.jpg')}}"/>
                        </div>
                        <div class="column" style="background-color:rgb(235, 235, 235);">
                            <p style="text-align: center;">Sauvola Thresholding</p>
                            <input type="image" name="submit" onclick="change('static/main/images/predict/201710370311000/gambar3.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/201710370311000/gambar3.jpg')}}"/>
                        </div>
                </div>  
            ...
```


### Select

Pada proses ini edit file `select.html`.

***simpan_disini/templates/nim/select.html***

```html
        <!-- EDIT START-->
            <form class="login100-form validate-form" action="{{parent_location}}{PRODUCT_ID}/pred_selects" method=post enctype=multipart/form-data>
        <!-- EDIT END -->
                <div class="row" style=" width: 29%; margin: auto; padding: 15px;" >
                    <div class="column" >
                        <div class="form-check form-check-inline">
                            <p style="font-weight: bold; color: black;">Select Model : </p>
                        </div>   
                    </div>
                    <div class="column" >
                        <div class="form-check form-check-inline"> 
                            <select name="select_model" id="selected_model">
                            <!-- EDIT START-->
                                <option value='model1'>Nama Model 1</option>
                                <option value='model2'>Nama Model 2</option>
                                <option value='model_js1'>Nama json Model 1</option>
                                <option value='model_js2'>Nama json Model 2</option>
                            <!-- EDIT END -->
                            </select>
                        </div>   
                    </div>
                </div>
            ...
            <!-- EDIT START-->
                <div class="row">
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="hidden" name="input_image" id="count" value="static/main/images/predict/{NIM}/gambar1.jpg" />
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar1.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar1.jpg')}}"/>
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar2.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar2.jpg')}}"/>
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar3.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar3.jpg')}}"/>
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar4.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar4.jpg')}}"/>
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar5.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar5.jpg')}}"/>
                    </div>
                </div>                    
                <div class="row">
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar6.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar6.jpg')}}"/>
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar7.jpg')"  style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar7.jpg')}}"/>
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar8.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar8.jpg')}}"/>
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar9.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar9.jpg')}}" />
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">{LABEL DATA}</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/{NIM}/gambar10.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/{NIM}/gambar10.jpg')}}"/>
                    </div>
                </div>
            <!-- EDIT END -->
            ...
```

Example:

```html
        <!-- EDIT START-->
            <form class="login100-form validate-form" action="{{parent_location}}11/pred_selects" method=post enctype=multipart/form-data>
        <!-- EDIT END -->
                <div class="row" style=" width: 29%; margin: auto; padding: 15px;" >
                    <div class="column" >
                        <div class="form-check form-check-inline">
                            <p style="font-weight: bold; color: black;">Select Model : </p>
                        </div>   
                    </div>
                    <div class="column" >
                        <div class="form-check form-check-inline"> 
                            <select name="select_model" id="selected_model">
                            <!-- EDIT START-->
                                <option value='model1'>ResNet 50</option>
                                <option value='model2'>BaseCNN</option>
                                <option value='model_js1'>ModelKeren JSON</option>
                                <option value='model_js2'>ModelSuperKeren JSON</option>
                            <!-- EDIT END -->
                            </select>
                        </div>   
                    </div>
                </div>
            ...
            <!-- EDIT START-->
                <div class="row">
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">Otsu Thresholding</p>
                        <input type="hidden" name="input_image" id="count" value="static/main/images/predict/201710370311000/gambar1.jpg" />
                        <input type="image" name="submit" onclick="change('static/main/images/predict/201710370311000/gambar1.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/201710370311000/gambar1.jpg')}}"/>
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">Niblack Thresholding</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/201710370311000/gambar2.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/201710370311000/gambar2.jpg')}}"/>
                    </div>
                    <div class="column" style="background-color:rgb(235, 235, 235);">
                        <p style="text-align: center;">Sauvola Thresholding</p>
                        <input type="image" name="submit" onclick="change('static/main/images/predict/201710370311000/gambar3.jpg')" style="height: 235px;width: 235px;padding-left: 10%;padding-right: 10%;padding-bottom: 10%;" src="{{ url_for('static', filename='main/images/predict/201710370311000/gambar3.jpg')}}"/>
                    </div>
                </div>                    
            ...
```

### Run Flask on Localhost

Sebelum di upload ke Web Riset lebih baik di run local terlebih dahulu untuk mengetahui apakah project sudah berjalan dengan baik atau masih ada bug.

```py
python simpan_disini/main_example.py
```
