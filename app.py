from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder="templates")

basedir = os.path.abspath(os.path.dirname(__file__))
filesFolder = os.path.join(basedir, 'files')
file1path = ""
file2path = ""

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/upload1', methods=['POST'])
def upload1():
    try:
        file1 = request.files['inputfile1']
        print(file1)
        global file1path
        file1path = os.path.join(filesFolder, 'unmasked.' + file1.filename.split('.')[-1])
        file1.save(file1path)
        return render_template('index.html', file1path=file1path, file2path=file2path)

    except:
        print('error upload')

@app.route('/upload2', methods=['POST'])
def upload2():
    try:
        file2 = request.files['inputfile2']
        print(file2)
        global file2path
        file2path = os.path.join(filesFolder, 'masked.' + file2.filename.split('.')[-1])
        file2.save(file2path)
        return render_template('index.html', file1path=file1path, file2path=file2path)

    except:
        print('error upload')

@app.route('/verify', methods=['GET'])
def verify():
    try:
        model = request.args.get('model')
        print(model)
        # result from 2 pics and model number
        result = '' + file1path + 'and' + file2path + '\nmodel=' + model
        # delete files uploaded
        cleanUpPic()
        return render_template('result.html', result=result)
    except Exception as e:
        print(e)
        return "??"

def cleanUpPic():
    for filename in os.listdir(filesFolder):
        file_path = os.path.join(filesFolder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            else:
                print('aba')
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            

if __name__ == '__main__':
   app.run()
