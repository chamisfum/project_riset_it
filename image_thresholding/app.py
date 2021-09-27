# web-app for API image manipulation
from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
import cv2
from werkzeug.utils import redirect
import Thresh as img_thres

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# default access page
@app.route("/")
def main():
    return render_template('proses.html')


# upload selected image and forward to processing page
@app.route("/proses1", methods=['POST'])
def proses1():
    target = os.path.join(APP_ROOT, 'static/images/')
    filename = request.files['file']
    print('filename : ',filename)

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # save file
    data = os.path.join(target, "query.jpg")
    filename.save(data)
    img = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
    print(img)
    # cek mode
    if 'otsu' in request.form.get('select_thresholding'):
        mode = 'otsu'
    elif 'niblack' in request.form.get('select_thresholding'):
        mode = 'niblack'
    elif 'sauvola' in request.form.get('select_thresholding'):
        mode = 'sauvola'
    #process
    if mode == 'otsu':
        print('start otsu')
        img_res = img_thres.otsu_thresh(img)
        print(img_res)
        cv2.imwrite("/".join([target, 'result.jpg']),img_res)
    elif mode == 'niblack':
        img_res = img_thres.niblack_thresh(img)
        cv2.imwrite("/".join([target, 'result.jpg']),img_res)
    elif mode == 'sauvola':
        img_res = img_thres.sauvola_thresh(img)
        cv2.imwrite("/".join([target, 'result.jpg']),img_res)

    # forward to processing page
    return redirect('/')

@app.route("/proses2", methods=["POST"])
def proses2():
    target = os.path.join(APP_ROOT, 'static/images/')
    filename = request.form.get('input_image')
    print('filename : ',filename)
    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)
    
    # save file
    img = Image.open(filename)
    img.save("/".join([target, filename]))
    img = cv2.imread("/".join([target, filename]), cv2.IMREAD_GRAYSCALE)
    input_img = img.copy()

    # cek mode
    if 'otsu' in request.form.get('select_thresholding'):
        mode = 'otsu'
    elif 'niblack' in request.form.get('select_thresholding'):
        mode = 'niblack'
    elif 'sauvola' in request.form.get('select_thresholding'):
        mode = 'sauvola'

    #process
    if mode == 'otsu':
        img_res = otsu_thresh(img)
        cv2.imwrite("/".join([target, 'result.jpg']),img)
    elif mode == 'niblack':
        img_res = niblack_thresh(img)
        cv2.imwrite("/".join([target, 'result.jpg']),img)
    elif mode == 'sauvola':
        img_res = sauvola_thresh(img)
        cv2.imwrite("/".join([target, 'result.jpg']),img)

    # forward to processing page
    return render_template("proses.html")


# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)

@app.route('/backup')
def coba():
    return render_template("backup.html")


if __name__ == "__main__":
    app.run(debug=True)

