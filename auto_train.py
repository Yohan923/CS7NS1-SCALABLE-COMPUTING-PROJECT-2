import os
import platform
import argparse
import subprocess

WINDOWS = 'windows'
LINUX = 'linux'
MAC = 'mac'
DARWIN = 'darwin'

WINDOWS_VENV = [
    'python -m venv venv',
]
WINDOWS_ACTIVATE = [
    '.\\venv\\Scripts\\activate'
]

UNIX_VENV = [
    'python3 -m venv ./venv'
]

UNIX_ACTIVATE = [
    '. ./venv/bin/activate'
]


SET_UP_ENV_COMMANDS = [
    'git pull',
    'pip install numpy==1.19.3',
    'pip install tensorflow==2.4.0rc1',
    'pip install opencv-python',
    'pip install captcha'
]
GEN_TRAIN_COMMANDS = [
    'python generate.py --width 128 --height 64 --length 1 --symbols symbols.txt --count 6144 --output-dir training-dataset', 
    'python generate.py --width 128 --height 64 --length 2 --symbols symbols.txt --count 6144 --output-dir training-dataset',
    'python generate.py --width 128 --height 64 --length 3 --symbols symbols.txt --count 6144 --output-dir training-dataset',
    'python generate.py --width 128 --height 64 --length 4 --symbols symbols.txt --count 6144 --output-dir training-dataset',
    'python generate.py --width 128 --height 64 --length 5 --symbols symbols.txt --count 6144 --output-dir training-dataset',
    'python generate.py --width 128 --height 64 --length 6 --symbols symbols.txt --count 6144 --output-dir training-dataset',
    'python generate.py --width 128 --height 64 --length 7 --symbols symbols.txt --count 6144 --output-dir training-dataset',
    'python generate.py --width 128 --height 64 --length 8 --symbols symbols.txt --count 6144 --output-dir training-dataset'
]

GEN_VALID_COMMANDS = [
    'python generate.py --width 128 --height 64 --length 1 --symbols symbols.txt --count 612 --output-dir validation-dataset',
    'python generate.py --width 128 --height 64 --length 2 --symbols symbols.txt --count 612 --output-dir validation-dataset',
    'python generate.py --width 128 --height 64 --length 3 --symbols symbols.txt --count 612 --output-dir validation-dataset',
    'python generate.py --width 128 --height 64 --length 4 --symbols symbols.txt --count 612 --output-dir validation-dataset',
    'python generate.py --width 128 --height 64 --length 5 --symbols symbols.txt --count 612 --output-dir validation-dataset',
    'python generate.py --width 128 --height 64 --length 6 --symbols symbols.txt --count 612 --output-dir validation-dataset',
    'python generate.py --width 128 --height 64 --length 7 --symbols symbols.txt --count 612 --output-dir validation-dataset',
    'python generate.py --width 128 --height 64 --length 8 --symbols symbols.txt --count 612 --output-dir validation-dataset'
]

TRAIN_COMMAND = [
    'python -u train.py --width 128 --height 64 --symbols symbols.txt --batch-size 32 --epochs 2 --output-model model --train-dataset training-dataset --validate-dataset validation-dataset --input-model model'
]

def run_commands(commands):
    for command in commands:
        print(command)
        os.system(command)
        


def create_venv(system):
    if system == WINDOWS:
        command = WINDOWS_VENV
    else:
        command = UNIX_VENV
    
    run_commands(command)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--system', help='the operating system script is running on (Windows, Linux, Mac)', type=str)
    args = parser.parse_args()
    
    if args.system:
        system = args.system.lower()
    else:
        system = platform.system()
        system = system.lower()

    if system not in [LINUX, MAC, DARWIN, WINDOWS]:
        print("doesn't seem to have a valid os specified")
        exit(1)

    if not os.path.exists("venv"):
        create_venv(system)

    commands = []

    if system == WINDOWS:
        activate_env_command = WINDOWS_ACTIVATE
    else:
        activate_env_command = UNIX_ACTIVATE

    commands += activate_env_command
    commands += SET_UP_ENV_COMMANDS
    
    if not os.path.exists("training-dataset"):
        commands += GEN_TRAIN_COMMANDS

    if not os.path.exists("validation-dataset"):
        commands += GEN_VALID_COMMANDS

    commands += TRAIN_COMMAND

    os.system(' && '.join(commands))

