from flask import Flask, render_template, request
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from PIL import Image
import os

app = Flask(__name__, template_folder="templates")

basedir = os.path.abspath(os.path.dirname(__file__))
filesFolder = os.path.join(basedir, 'files')
file1path = ""
file2path = ""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
mtcnn = MTCNN(
    image_size=160,
    margin=14,
    selection_method='center_weighted_size'
)
pretrained_model = InceptionResnetV1(
    classify=False,
    pretrained='vggface2'
).eval()
real_world_model = InceptionResnetV1(classify=False)
real_world_model.load_state_dict(torch.load("models\mixed_mask_triplet", map_location=device), strict=False)
real_world_model.eval()
lfw_model = InceptionResnetV1(classify=False)
lfw_model.load_state_dict(torch.load("models\lfw_mask_triplet", map_location=device), strict=False)
lfw_model.eval()

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
        is_same = True
        if model == "Pretrained Baseline":
            is_same = compare(pretrained_model, 0.3)
        elif mode == "Transfer Learning LFW":
            is_same = compare(lfw_model, 0.3)
        else:
            is_same = compare(real_world_model, 0.3)
        print(model)
        # result from 2 pics and model number
        result = '' + file1path + ' and ' + file2path + '\nmodel=' + model
        if is_same:
            result += " are the same person"
        else:
            result += " are not the same person"
        # delete files uploaded

        return render_template('result.html', result=result, file1path=('/files/' + file1path.split('\\')[-1]), file2path=file2path)
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

def compare(model, threshold):
    # Image Preprocessing
    img1 = Image.open(path1)
    img1 = mtcnn(img1)
    img2 = Image.open(path2)
    img2 = mtcnn(img2)
    img1 = img1.convert('RGB')
    img1 = trans(img1).to(device)
    img2 = img2.convert('RGB')
    img2 = trans(img2).to(device)
    img1_embedding = resnet(img1.unsqueeze(0)).detach().numpy()
    img2_embedding = resnet(img2.unsqueeze(0)).detach().numpy()
    if distance(img1_embedding, img2_embedding) <= threshold:
        return True
    else:
        return False

if __name__ == '__main__':
    ondebug = True
    # ondebug = False
    app.run(debug=ondebug)
