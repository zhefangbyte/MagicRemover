# Introduction
This is a graduate project for my undergraduate study. 

It aims to build an app that can be used to colour and repair images. Also, it includes some basic functions that an image editor will do.

See the brief intro video via [Bilibili](https://www.bilibili.com/video/BV13m421W7Zr/) in Chinese or [YouTube](https://www.youtube.com/watch?v=j_cjcSy7R-w) in English.

# Development
By default, this project uses PyTorch packages with CPU versions, 
just in case you want to make it back to the CPU version,
run the following command line:
```commandline
pip install --force-reinstall torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu
```

For GPU development, reinstall them by running:
```commandline
pip install --force-reinstall torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

The GPU version was tested successfully on NVIDIA GTX 1660 SUPER.

Look at [pytorch official website](https://pytorch.org/get-started/previous-versions/) for more detail.

Note that package `pillow` may be reinstalled with a new version 
which can lead to an error while launching the app 
after changing the torch version. To stop it from happening, 
use the following command line to reinstall `pillow`:
```commandline
pip install --force-reinstall pillow==9.5.0
```
