# 1 Installation

```shell
git clone --recursive git@github.com:SergeyuiL/top-down-map.git
cd top-down-map/
conda env create -f environment.yaml
conda activate tdmap
python -m spacy download en_core_web_sm

mkdir ckp && cd ckp
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

cd segment-anything; pip install -e .; cd ..
```

# 2 Data preparation

Feel free to download my rosbag:[indoor rosbag](https://drive.google.com/file/d/1t72MNzk0BFzAl7X1dX_5jvwkef4IPqyT/view?usp=sharing)

Preprocess:[rosbag play](https://github.com/SergeyuiL/rosbag_play.git)

Save the data in advance to the data folder in the following format：

```shell
data
├── camera_info.json
├── color
│   ├── 0.png
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   ├── ...
├── depth
│   ├── 0.png
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   ├── ...
└── pose
    ├── 0.txt
    ├── 1.txt
    ├── 2.txt
    ├── 3.txt
    ├── 4.txt
    ├── ...
```

# 3 segment and filter expected semantic classes

```shell
python Semantic-Segment-Anything/scripts/main_ssa_engine.py --data_dir=<The name of your dataset> --out_dir=output --world_size=1 --save_img --sam --ckpt_path=ckp/sam_vit_h_4b8939.pth --light_mode
# e.g. python Semantic-Segment-Anything/scripts/main_ssa_engine.py --data_dir=data/color --out_dir=data/semantic --world_size=1 --save_img --sam --ckpt_path=ckp/sam_vit_h_4b8939.pth --light_mode

python scripts/seg_my_classes.py --data_dir=<path to data> 
# e.g. python scripts/seg_my_classes.py --data_dir=data/ 
```

# 4 create top-down color and seg map

```shell
python scripts/mapping_utils.py --data_dir=<path to data> --map_save_dir=<path to save maps>
# e.g., python scripts/mapping_utils.py --data_dir=data/ --map_save_dir=maps/locobot

# show color map
# python scripts/show_map.py --map_save_dir=maps/locobot
```


