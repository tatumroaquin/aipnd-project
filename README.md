# Flower classifier AI

### Steps to install packages

1. [Install Conda](https://docs.anaconda.com/miniconda/)

__Linux__
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

__Windows__
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" miniconda.exe /S
del miniconda.exe
```

2. [Install PyTorch](https://pytorch.org/get-started/locally/)
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

3. Install numpy, matplotlib
```
conda install numpy matplotlib
```

### Steps to train

1. Download and extract the [flowers_data.tar.gz](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) to the root of the project

1. Activate anaconda or miniconda, to allow shell access to our python packages
```
conda activate
```

1. Train with defaults
```
python train.py flowers
```

1. Train with specific epoch
```
python train.py --epoch 5 flowers
```

1. Train with a different architecture
```
python train.py --arch resnet flowers
python train.py --arch vgg flowers
```

1. Train with different number of hidden units
```
python train.py --hidden_units 256 flowers
```

1. Train without GPU
```
python train.py --no-gpu flowers
python train.py --no-gpu flowers
```

### Steps to predict
1. Predict the class of an image with (default is VGG Model)
```
python predict.py flowers/test/10/image_07090.jpg checkpoint.pth
```

1. Predict top-k classes of one images
```
python predict.py --top_k 5 flowers/test/10/image_07090.jpg checkpoint.pth
```

1. Load a custom category mapping JSON
```
python predict.py --category_names cat_to_name.json flowers/test/10/image_07090.jpg checkpoint.pth
```
