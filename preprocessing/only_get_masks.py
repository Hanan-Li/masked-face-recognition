import os
import shutil
from pathlib import Path
from PIL import Image

base_path = "C:\\Users\\david\\Documents\\masked-face-recognition"
masked_path = os.path.join(base_path, "AFDB_masked_face_dataset")
nonmasked_path = os.path.join(base_path, "AFDB_face_dataset")
mixed_path = os.path.join(base_path, "mixed_face_subset")
# if not os.path.isdir(mixed_path):
#     os.mkdir(mixed_path)

entries = os.listdir(masked_path)

for entry in entries:
    src = os.path.join(masked_path, entry)
    dst = os.path.join(mixed_path, entry)
    shutil.copytree(src, dst)
    # for idx, img in enumerate(os.listdir(dst)):
    #     dst_filename = os.path.join(dst, img)
    #     new_name = os.path.join(dst, "masked_" + str(idx))
    #     os.rename(dst_filename, new_name)
    non_masked_dir = os.path.join(nonmasked_path, entry)
    nonmasked_entries = os.listdir(non_masked_dir)
    for i in range(5):
        src_filename = os.path.join(non_masked_dir, nonmasked_entries[i])
        dst_filename = os.path.join(dst, nonmasked_entries[i])
        shutil.copyfile(src_filename, dst_filename)


mixed_entries = os.listdir(mixed_path)
for entry in mixed_entries:
    src = os.path.join(mixed_path, entry)
    for entry in os.listdir(src):
        img_path = os.path.join(src, entry)
        if entry == ".DS_Store":
            os.remove(img_path)
        else:
            img = Image.open(img_path)
            img = img.resize((160,160))
            img.save(img_path)