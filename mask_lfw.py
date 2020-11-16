import os
import subprocess

import shutil
from pathlib import Path
from PIL import Image

base_path = "C:\\Users\\david\\Documents\\masked-face-recognition\\lfw_complete"
lfw_orig_path = os.path.join(base_path, "lfw")
mask_the_face_path = "C:\\Users\\david\\Documents\\MaskTheFace"
entries = os.listdir(lfw_orig_path)
for entry in entries:
    folder_path = os.path.join(lfw_orig_path, entry)    
    command = 'cd C:\\Users\\david\\Documents\\MaskTheFace && python mask_the_face.py --path {} --mask_type random --verbose --write_original_image'.format(folder_path)
    print(command)
    subprocess.call(command, shell=True)

