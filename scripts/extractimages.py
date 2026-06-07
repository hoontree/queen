import os
import sys
import shutil

folder_path = sys.argv[1]

colmap_path = "./colmap_tmp"
images_path = os.path.join(colmap_path, "images")
os.makedirs(images_path, exist_ok=True)

dir1 = os.path.join("data", folder_path)
i = 0
for folder_name in sorted(os.listdir(dir1)):
    cam_dir = os.path.join(dir1, folder_name, "images")
    if not os.path.isdir(cam_dir):
        continue
    src_path = os.path.join(cam_dir, "0000.png")
    if not os.path.isfile(src_path):
        continue
    i += 1
    dst_path = os.path.join(images_path, f"image{i}.jpg")
    shutil.copyfile(src_path, dst_path)

print("End!")
