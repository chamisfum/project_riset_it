# web-app for API image manipulation

from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
import cv2
from image_thresholding import otsu_thresh, niblack_thresh, sauvola_thresh

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# default access page
@app.route("/")
def main():
    return render_template('index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)

    # forward to processing page
    return render_template("processing.html", image_name=filename)

@app.route("/threshold", methods=["POST"])
def threshold():

    # retrieve parameters from html form
    if 'otsu' in request.form['mode']:
        mode = 'otsu'
    elif 'niblack' in request.form['mode']:
        mode = 'niblack'
    elif 'sauvola' in request.form['mode']:
        mode = 'sauvola'
    else:
        return render_template("error.html", message="Mode not supported (vertical - horizontal)"), 400
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    # img = Image.open(destination)
    img = cv2.imread(destination, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (224, 224))
    res = img.copy()

    if mode == 'otsu':
        # img = cv2.imread(destination, 0)
        img = otsu_thresh(img)
    elif mode == 'niblack':
        # img = cv2.imread(destination, cv2.IMREAD_GRAYSCALE)
        img = niblack_thresh(img)
    elif mode == 'sauvola':
        # img = img.convert('LA')
        img = sauvola_thresh(img)

    else:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # save and return image
    destination = "/".join([target, 'temp.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    # img.save(destination)
    cv2.imwrite(destination, img)

    return send_image('temp.png')


# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run(debug=True)

