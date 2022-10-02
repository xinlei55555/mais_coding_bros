from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
import base64
import io
import os
import time

app = Flask(__name__, static_folder='F:\Hackathon\mais_coding_bros\web_dev\static')
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'static')
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['UPLOAD_PATH'] = filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/myform.cgi', methods=['POST'])
def count():
    print('done')
    uploaded_file = request.files['fileupload']
    print(uploaded_file)
    #absolute_path = os.path.abspath(app.config['UPLOAD_PATH']+'image.png')

    new_name = "image" + str(time.time()) + ".png"
    uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], new_name))
    print('saved file')


    return render_template('result.html', result='Healthy', image_name = new_name)

if __name__ == '__main__':
    app.run(debug=True)