<h1>Automation Scripts</h1>

Before running scripts please ensure you are running from within the root directory of the project e.g ./CS7NS1-SCALABLE-COMPUTING-PROJECT-2 and have python 3.6 or above installed.

<h2>Training</h2>

---
[auto_train.py](./auto_train.py)

- Optional input --system should be provided if the script cannot determine the operating system that you are running this script on. (windows, linux, mac). The script might fail if you are running on incompatible systems.

**This script should not be run on the Pi as it installs tensorflow which requires 64-bit systems.**  

To run the script, open command line tool, ```cd``` into project root directory, and run [auto_train.py](./auto_train.py) using python. If you want to train using GPU, you will need to install CUDA toolkit 11.0 and cuDNN 8.0.2 with compatible GPU for tensorflow 2.4.0. The program is set to use CPU if GPU can not be found on the machine

The script will execute the the following steps

1. The script determines if python ```venv``` was created, which should not be when first pulled from the repo. It will create a ```venv``` in the root directory
2. The script should activate the ```venv```
3. ```git pull``` to ensure using the latest version of code
4. ```pip install numpy==1.19.3``` install numpy via pip
5. ```pip install tensorflow==2.4.0rc1``` install tensorflow via pip
6. ```pip install opencv-python``` install opencv via pip
7. ```pip install captcha``` install captcha package for generating images
8. generate a set of training images in a directory named ```training-dataset``` if this directory does not exist. The setup currently is 8 sets of 6144 images containing 1-8 characters each. Please ensure this directory is deleted if you want to generate fresh sets of images
9. generate a set of validation images in a directory named ```validation-dataset``` if this directory does not exist. The setup currently is 8 sets of 612 images containing 1-8 characters each. Please ensure this directory is deleted if you want to generate fresh sets of images
10. runs [train.py](./train.py) with default hard coded arguments, arguments can be changed by modifying line 57 of the script. By default this will continue training the existing model ```model.h5``` provided with the code bundle. Delete all files with name ```model``` to train new models or remove the input argument ```--input-model```

With the current setup, you should see the training running for 100 epoch with batch size of 32. Model is saved after every epoch so we do not lose progress in unexpected circumstances. The model is converted to ```tflite``` model only after the training ends, please ensure to continue training on the model for at least 1 more epoch until training properly ends if the training ends unexpectedly to enure ```tflite``` model is trained.
After finish training the model, you can transfer the model to the PI using ```scp``` if you wish as you will not be able to upload model to GitHub without password.

<h2>Classification</h2>

---

[auto_classify.sh](./auto_classify.sh)


To run the script, open command line tool on the Pi, ```cd``` into project root directory, and run [auto_classify.py](./auto_classify.py) using the command

    source ./auto_classify.sh

```source``` is important as you will not activate the ```venv``` if you do not use source to run the script. If you get an error ```Permision Denied``` try run the following command

    chmod +x ./run.sh

The script will execute the the following steps

1. The script determines if python ```venv``` was created, which should not be when first pulled from the repo. It will create a ```venv``` in the root directory
2. set ```LD_LIBRARY_PATH```, this only works on my own Pi as I installed OpenBlas and numpy from source. It is needed to run numpy there but can be ignored on other machines
2. The script should activate the ```venv```
3. ```pip install numpy``` install numpy via pip
4. ```pip install opencv-python``` install opencv via pip
5. install tflite_runtime-2.5.0 for 32-bit arm via pip
6. ```git pull``` to ensure using the latest version of code
7. run classification using [classify.py](./classify.py)

By default, the script will classify images stored in the directory ```./captchas/``` and store the results in ```stuff.txt```. You can change this by modifying the script at line 12. 


<h2>Misc</h2>

---

[send.sh](./send.sh)
not important, just used for my own convenience to ```scp``` result text file to my own PC