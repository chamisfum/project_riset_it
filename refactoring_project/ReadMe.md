# Clean Code IRC Deployment

## Motivation and Brief

<style>Project ini menyediakan package module python untuk melakukan deployment model `Deep Learning Image Classification` menggunakan Flask micro framework. Goals utama dari project ini adalah menyediakan arsitektur aplikasi flask yang bersih dan mudah untuk dimplementasikan untuk project machine learning maupun deep learning. Clean Code arsitektur menjadi motivasi utama untuk membangun project ini. Sehingga dalam project ini kami berusaha semaksimal mungkin untuk dapat menerapkan arsitektur Clean Code sebagai reverensi silahkan baca dokumentasi Clean Code [disini](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html){text-align: justify}</style>

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

## Prerequisite 

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