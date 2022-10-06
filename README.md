## Adherent mist and raindrop removal from a single image using attentive convolutional network

Da He, Xiaoyu Shang, Jiajia Luo \
[Paper link](https://www.sciencedirect.com/science/article/abs/pii/S0925231222008918)
 - - -
### Citation:
If you find this implementation, the article, or our dataset is helpful / useful / inspiring, please cite the following :D

 - - -
This repository is the official implementation.  

The code implementation is built based on the repository https://github.com/zhilin007/FFA-Net.  

### Dependencies
* python3
* PyTorch>=1.0
* NVIDIA GPU+CUDA
* numpy, matplotlib, Pillow

### Dataset preparation and description
Download link: [Dropbox](https://drive.google.com/file/d/1RFERNR4Jp-kHjnE4e_5pmwqGComIDoRR/view?usp=sharing), [Baidu Yun 百度云](https://pan.baidu.com/s/1c_JD9DmnzHF4N4OuCn3jow)(password: fgb5)

To use the dataset with this implementation, decompress the `MistAndRaindrop_dataset.zip` file and place the obtained folder `MistAndRaindrop` in `./datasets`.  

There are 3 sub-sets (`train`, `val`, `test`). In each of the sub-set, each image file is named as "XXX.YY-TYPE.png". Images with the same XXX correspond to the same scene. For each scene, the filename containing "landscape" indicates the Ground Truth (clean) image, while other filenames correspond to the degraded images with differently and randomly adherent mist / raindrops.

For example, in the `val` sub-set, scene `611` contains three image files: `611.0-landscape.png`, `611.0-spray.png`, `611.1-spray.png`. The file `611.0-landscape.png` is the Graond Truth file, corresponding to two degraded images.

### Usage

#### Test
```
python test.py
```

### Samples
![degraded_sample_1](/samples/405.0-spray.png) ![restored_sample_1](/samples/405.0-spray_res.png)  
![degraded_sample_2](/samples/443.3-spray.png) ![restored_sample_2](/samples/443.3-spray_res.png)  


