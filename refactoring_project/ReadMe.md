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

Happy Coding :)
