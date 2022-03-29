import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/train --output_dir ../data/DIV2K/DBPN/train --image_size 370 --step 185 --num_workers 10")
os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/valid --output_dir ../data/DIV2K/DBPN/valid --image_size 370 --step 185 --num_workers 10")
