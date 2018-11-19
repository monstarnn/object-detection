from config import build_config
from subprocess import check_output
from argparse import ArgumentParser
import sys


def main():
    build_config()
    # "python object_detection/model_main.py --model_dir $TRAINING_DIR/$BUILD_ID --pipeline_config_path $SRC_DIR/faster_rcnn_resnet101_pets.config --num_train_steps=1000"
    parser = ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--research_dir')
    # parser.add_argument('--num_steps', default=50000)
    args, _ = parser.parse_known_args()


    targs = sys.argv[:]
    targs[0] = args.research_dir + '/object_detection/model_main.py'
    targs.insert(0, sys.executable or 'python')
    print(check_output(targs))


if __name__ == '__main__':
    main()
