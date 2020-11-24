if [ ! -d "venv/" ]; then
	python3 -m venv ./venv
fi
source venv/bin/activate
if [ -d "/users/ugrad/zhangti/project2/OpenBLAS/lib/" ]; then
	export LD_LIBRARY_PATH=/users/ugrad/zhangti/project2/OpenBLAS/lib/
fi
pip3 install numpy
pip3 install opencv-python
pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp38-cp38-linux_armv7l.whl
git pull
python3 classify.py --model-name model --captcha-dir ./captchas/ --output ./stuff.txt --symbols symbols.txt
