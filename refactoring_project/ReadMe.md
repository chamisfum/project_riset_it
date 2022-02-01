## Dokumentasi

Dalam kode ini telah disediakan kode untuk deploy image classification menggunakan flask (loacal / rilis) dengan menggunakan kode ini kalian hanya perlu import paket `src.config` untuk melakukan preprocessing gambar RGB dan Grayscale, Predict Model Sigmoid dan Softmax, Compare dan Select Model, Select Gambar Query.

```
from src import config 
""" import paket untuk klasifikasi : preprocessing RGB dan Grayscale, 
Predict Model Sigmoid dan Softmax, Compare dan Select Model, Select Gambar
"""

class_list     = {'GLIOMA': 0, 'MENINGIOMA': 1, 'PITUITARY': 2} 
""" Ganti dengan nama kelas data yang urutannya mengikuti hukum abjad 
    (key = nama kelas, value = index kelas dalam data generator)
"""
path           = "static/model/" # path lokasi menyimpan model
queryImagePath = "static/queryImage/" # path lokasi penyimpanan contoh query image
```

Ganti nama model dan gambar query dengan pola penamaan sbb :
- nama query gambar `namaKelas_namaGambar.jpg` (gambar boleh berekstensi jpg, png, jpeg, grayscale atau RGB)
- nama model `namaModel_model.h5` atau `namaModel_model.hdf5` (untuk hdf5 model) atau `namaModel_model.json` (untuk model json) 
- nama bobot (weight) `nameWeight_weight.h5`

**Untuk Sigmoid model silahkan hapus bagian tabel performa di template html**


Happy Coding :)


# Documentation

This part is an application for deploying image classification using Deep Learning algorithm. 
Use this part as main service endpoint for managing raw data from frontend into service backend. 
It would provide a collection of data such (prediction class, metric performance etc) 
from service backend into frontend.

READ ME FIRST !!

you are only need to change line of code with (# TO CHANGE) without () marking line. 
And you only need to change the app.py file as long as your project is Deeplearning for image classification.
You can change, add or remove any function each layer as you need (as long as it's not affect any service an attributes in this project)
To get better understanding of the whole function that might be you need please kindly check and read the documentation first!

TO CHANGE FILE

Beside you change any line of code in this application layer, you were also need to change some files like models, weights, and query images sample

You can find all models and weights in /static/model/ folder. Just delete any file in this folder and change with yours (models and weights *if any)
Rename your models an weight by adding the current name with "_model" and "_weight" without "" after the current name 

FOR EXAMPLE : 
            * current name : VGG19.h5, VGG19.json, VGG19bobot.h5
            * new name     : VGG19_model.h5, VGG19_model.json, VGG19bobot_weight.h5

You can find all query images in /static/queryImage/ folder. Just delete any file in this folder and change with your query images sample
Rename your query images by adding each current query image name with this pattern "<ClassName_><currentImageName>.<currentImageExtention>"
Change the <ClassName_> without <> or ""

FOR EXAMPLE : 
            * current name : 410.jpg, 450.png, 110.jpeg
            * new name     : Glioma_410.jpg, Meningioma_450.png, Pituitary_110.jpeg 

@cham_is_fum