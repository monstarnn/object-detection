from config import build_config
from subprocess import check_output
from argparse import ArgumentParser
import sys


def main():
    build_config()
    parser = ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--research_dir')
    # parser.add_argument('--num_steps', default=50000)
    args, _ = parser.parse_known_args()

    print(args)

    targs = sys.argv[:]
    print(targs)
    return

    targs[0] = args.research_dir + '/object_detection/export_inference_graph.py'
    targs.insert(0, sys.executable or 'python')
    targs.append("--pipeline_config_path")
    targs.append("faster_rcnn_resnet101_pets.config")

    targs.append("--trained_checkpoint_prefix")
    targs.append("$TRAINING_DIR/$BUILD/model.ckpt-${CHECKPOINT}")

    print(check_output(targs))


if __name__ == '__main__':
    main()
