# Introduction
This is a graduate project for SE at DGTI, HBUT. 
Its aim is to build an app, which can be used to colorize, repair images.
Also, it includes some basic functions as an image editor will do.

# Development
By default, this project use pytorch packages with CPU version, 
just in case if you want to make it back to CPU version,
run the following command line:
```commandline
pip install --force-reinstall torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu
```

For GPU development, reinstall them by running:
```commandline
pip install --force-reinstall torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

The GPU version was tested successfully on NVIDIA GTX 1660 SUPER.

Take a look at [pytorch official website](https://pytorch.org/get-started/previous-versions/) for more detail.

Note that package `pillow` may be reinstalled with a new version 
which can lead to an error while launching the app 
after changing the torch version. To stop it from happening, 
use the following command line to reinstall `pillow`:
```commandline
pip install --force-reinstall pillow==9.5.0
```
