# Project Documentation
**Clean Code Implementation**


## Motivation and Brief

Project ini menyediakan package module python untuk melakukan deployment model `Deep Learning Image Classification` menggunakan Flask micro framework. Goals utama dari project ini adalah menyediakan arsitektur aplikasi flask yang bersih dan mudah untuk dimplementasikan untuk project machine learning maupun deep learning. Clean Code arsitektur menjadi motivasi utama untuk membangun project ini. Sehingga dalam project ini kami berusaha semaksimal mungkin untuk dapat menerapkan arsitektur Clean Code sebagai reverensi silahkan baca dokumentasi Clean Code [disini](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)


## Structure

```html

refactoring_project
├───src
│   ├───config
│   │   ├───__init__.py
│   │   └───config.py
│   ├───infra
│   │   ├───__init__.py
│   │   └───infra.py
│   ├───service
│   │   ├───__init__.py
│   │   └───service.py
│   └───__init__.py
├───static
│   ├───model
│   │   ├───ExampleA_model.h5
│   │   ├───ExampleB_weight.h5
│   │   └───ExampleB_model.json
│   ├───queryImage
│   │   ├───ClassA_1.jpg
│   │   ├───ClassB_2.jpg
│   │   └───ClassC_3.jpg
│   └───queryUpload
│       └───temp.jpg
├───templates
│   ├───base.html
│   ├───base2.html
│   ├───compare.html
│   ├───result_compare.html
│   ├───result_select.html
│   └───select.html
├───app.py
└───requirements.txt
```


## Description

* `src`                berisi seluruh fungsi dan service utama pada aplikasi
* `src/infra`          merupakan folder penyimpanan layanan fungsi `infrastructure layer` yang terdiri dari barisan fungsi yang menyediakan layanan micro untuk setiap proses yang diperlukan (berisi fungsi sederhana yang hanya dapat melakukan sebuah tugas spesifik tertentu)
* `src/config`         merupakan folder penyimpanan layanan fungsi `configuration layer` yang terdiri dari barisan fungsi yang berperan sebagai jembatan antara `infrastructure layer` dan `service layer`. (helper layer)
* `src/service`        merupakan folder penyimpanan layanan fungsi `service layer` yang terdiri dari barisan fungsi yang menyediakan service atau layanan kompleks tertentu yang akan digunakan oleh `application layer` untuk mengolah dan mendapatkan datanya.
* `static/model`       berisi seluruh model dan bobot yang digunakan dalam aplikasi
* `static/queryImage`  berisi seluruh contoh gambar query untuk prediksi (setiap kelas data minimal terwakili 1 gambar yang tersimpan dalam folder ini)
* `static/queryUpload` berisi temporary uplaod gambar query yang akan diprediksi 
* `app.py`             `application layer` yang bertugas sebagai routing dan perantara user interface (UI) atau antrmuka pengguna dengan backend atau `service layer`.
* `requirements.txt`   daftar package python utama yang digunakan dalam applikasi anda


## File Naming

Anda dapat menemukan semua model dan bobot dalam folder `/static/model/`. Hapus saja file apa pun di folder ini dan ubah dengan milik Anda (model dan bobot *jika ada)
Ubah nama model dan bobot anda dengan menambahkan pola nama yang telah ditentukan pada nama model dan bobot yang anda miliki saat ini (tambahkan pola nama "_model" pada bagian nama model anda sebelum tanda "." extensi model dan tambahkan pola nama "_weight" pada bagian nama bobot anda sebelum tanda "." extensi bobot seluruhnya tanpa "")

<br>

### Contoh Penerapan :
* Nama sebenarnya saat ini                      : VGG19.h5, VGG19.json, VGG19bobot.h5
* Nama baru (setalah ditambahkan pola nama)     : VGG19_model.h5, VGG19_model.json, VGG19bobot_weight.h5

Anda dapat menemukan semua gambar query dalam folder `/static/queryImage/`. Hapus saja file apa pun di folder ini dan ubah dengan sampel gambar query Anda.
Ganti nama gambar query Anda dengan menambahkan setiap nama gambar query saat ini dengan mengikuti pola berikut `<ClassName_><currentImageName>.<currentImageExtention>`

* Ubah `<ClassName_>` tanpa `<>` dengan Nama class data dari masing - masing gambar.
* Untuk bagian `<currentImageName>.<currentImageExtention>` **tidak perlu diubah**
* Disini anda hanya perlu menambahkan nama kelas `<ClassName_>` pada bagian depan nama masing - masing nama file.
* Pastikan tidak ada karakter spasi dalam nama file baru.

<br>

### Contoh Penerapan :
* Nama sebenarnya saat ini                      : 410.jpg, 450.png, 110.jpeg
* Nama baru (setalah ditambahkan pola nama)     : Glioma_410.jpg, Meningioma_450.png, Pituitary_110.jpeg 


## First Time Running

* Install [python 3.7](https://www.python.org/downloads/release/python-370/) or letter
* Make virtualenv 
    * `pip install virtualenv`
    * `virtualenv [name of your new virtual environment]`
    * `cd [name of your new virtual environment]`
    * `source bin/activate`
    * `cd ..`
    * `cd refactoring_project`
* Install python package
    * `pip install -r requirements.txt`
* Run the `app.py`
    * `python app.py`
* Check on your web browser and try to access this ip [`127.0.0.1:5000`](http://127.0.0.1:5000)


## Local Setup

Perhatikan bahwa disini anda hanya perlu mengubah beberapa baris kode pada application layer `app.py`. Tanpa harus mengubah kode lain. **Hanya untuk deep learning image classification**. Untuk fungsi atau service yang belum tersedia maka anda bebas untuk menambahkannya sendiri di dalam project ini. Cukup ajukan pull request dengan menyertakan dokumentasi lengkap dari service atau function yang anda buat. Pastikan anda menerapkan arsitektur clean code sesuai dengan yang telah dibangun dalam project ini. Berikut contoh baris code yang perlu anda ubah ketika ingin menggunakan arsitektur project ini pada aplikasi `image classification` anda sendiri lihat snip code dibawah ini. Secara garis besar anda hanya perlu mengganti baris kode bertanda `# TO CHANGE` saja dalam `app.py`.

* Bagian ini digunakan untuk melakukan prediksi dan preprocessing gambar RGB / Grayscale. Pilih salah satu service yang sesuai dengan data gambar yang anda gunakan. Buka komentar bagian yang anda gunakan, anda juga dapat menghapus atau menambahkan komentar pada bagian yang tidak anda gunakan.

```python

""" Uncomment to use this part if you using RGB imgae as input prediction"""
PredictRGBImageList             = service.PredictInputRGBImageList  # TO CHANGE 
PredicRGBImage                  = service.PredictInputRGBImage  # TO CHANGE 

""" Uncomment to use this part if you using grayscale imgae as input prediction"""
# PredictGrayImageList            = service.PredictInputGrayImageList  # TO CHANGE 
# PredicGrayImage                 = service.PredictInputGrayImage  # TO CHANGE 

```

* Pada bagian ini ganti class dictionary sesuai dengan data yang anda miliki. Dalam hal ini class name disimpan sebagai key of dictionary dan nomor urut class disimpan dalam value (nomor urut dimulai dari 0). Class name urut berdasarkan abjad lihat flow_from_directory.

```python

CLASS_DICT          = {'GLIOMA': 0, 'MENINGIOMA': 1, 'PITUITARY': 2} # TO CHANGE

```

* Ganti dengan product information anda dari product yang anda uplaod dalam web riset.informatika.umm.ac.id

```python

"""
IMPORTANT!
please change this part into your product detail and configuration
"""
# TO CHANGE -> start
PARENT_LOCATION     = "data_science_product" # TO CHANGE # represent to parent of project web service configuration 
TOPIC_NAME          = "Brain Tumor Disease"  # TO CHANGE # represent topic name 
AREA_OF_INTEREST_ID = "1"                    # TO CHANGE # represent area of interest id 
TOPIC_ID            = "1"                    # TO CHANGE # represent topic id 
PRODUCT_ID          = "1"                    # TO CHANGE # represent product id 
# TO CHANGE -> end

```

* Untuk konfigurasi local tidak perlu ganti bagian ini

```python

"""
LOCAL CONFIG!
    Comment this part before releasing your application in production
"""
app = Flask(__name__) # TO CHANGE 

```

* Pada bagian ini pilih salah satu service yang sesuai dengan jenis data citra yang anda miliki. `PredictRGBImageList` untuk data RGB dan `PredictGrayImageList` untuk data grayscale. Tambahkan Comment untuk service yang tidak anda pakai atau anda juga bisa menghapusnya. Pastikan anda sudah menggunakan salah satu service ini untuk menjalankan aplikasi anda (uncomment untuk menggunakan service)

```python

    predictionResult, predictionTime = PredictRGBImageList(choosenModelList, MODEL_PATH, getImageFile)  # TO CHANGE 
    # predictionResult, predictionTime = PredictGrayImageList(choosenModelList, MODEL_PATH, getImageFile)  # TO CHANGE 

```

* Untuk konfigurasi local gunakan bagian ini (tidak perlu diubah hanya pastikan bagian `app.run(debug=True, host='127.0.0.1', port=5000)` telah aktif / di uncomment.

```python

    # LOCAL DEVELOPMENT CONFIG
    app.run(debug=True, host='127.0.0.1', port=5000) # TO CHANGE 

```


## Production Setup

* Untuk konfigurasi di production ganti bagian `app = Flask(__name__) # TO CHANGE ` dengan snip code dibawah ini:

```python

"""
PRODUCTION CONFIG!
    uncomment and change the static_url_path to into url project path
"""
# app = Flask(__name__, static_url_path=PARENT_LOCATION+'static') # TO CHANGE 

```

* Untuk konfigurasi di production uncomment bagian snip code dibawah dan hapus atau comment bagian `app.run(debug=True, host='127.0.0.1', port=5000) # TO CHANGE `. Pada bagian snip cod ini anda juga dapat menambahkan ssl cert & key yang valid untuk domain dan sub domain anda di production.

```python

    # PRODUCTION CONFIG    
    # app.run(debug=False, host='0.0.0.0', port=2000, # TO CHANGE 
    # HANDLE SSL CERT AND KEYS
    #         ssl_context = ('/home/admin/conf/web/ssl.riset.informatika.umm.ac.id.crt', # TO CHANGE 
    #                       '/home/admin/conf/web/ssl.riset.informatika.umm.ac.id.key')) # TO CHANGE 
```

## Credit

Sebagai bentuk kontribusi untuk keberlanjutan project ini seluruh saran yang membangun, kontribusi, bug report dan segala bentuk optimasi sistem dapat mengajukan PR (pull request) dengan menyertakan documentasi lengkap atau deskripsi fitur baru.

[@cham_is_fum](https://github.com/chamisfum)