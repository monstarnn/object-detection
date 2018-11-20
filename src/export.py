from config import build_config
from subprocess import check_output
from argparse import ArgumentParser
import sys

from mlboardclient.api import client


def main():
    build_config()
    parser = ArgumentParser()
    parser.add_argument('--research_dir')
    parser.add_argument('--training_dir')
    parser.add_argument('--build')
    parser.add_argument('--num_steps')
    # parser.add_argument('--num_steps', default=50000)
    args, _ = parser.parse_known_args()

    print(args)

    targs = sys.argv[:]

    targs[0] = args.research_dir + '/object_detection/export_inference_graph.py'
    targs.insert(0, sys.executable or 'python')

    targs.append("--pipeline_config_path")
    targs.append("faster_rcnn_resnet101_pets.config")

    targs.append("--trained_checkpoint_prefix")
    targs.append("%s/%s/model.ckpt-%s" % (args.training_dir, args.build, args.num_steps))

    targs.append("--output_directory")
    targs.append("%s/model/%s" % (args.training_dir, args.build))

    print(check_output(targs))

    print(targs)
    return

    m = client.Client()
    m.model_upload('$MODEL_NAME', '$MODEL_VERSION', '$TRAINING_DIR/model/$BUILD/saved_model')
    m.update_task_info({'model': '#/$WORKSPACE_NAME/catalog/mlmodel/$MODEL_NAME/versions/$MODEL_VERSION'})

    # BUILD=1; CHECKPOINT=1000; MODEL_NAME=object-detection; MODEL_VERSION=1.0.0;
    # sed -i -e "s|%DATADIR%|$DATA_DIR|g" $SRC_DIR/faster_rcnn_resnet101_pets.config;
    # python object_detection/export_inference_graph.py
    # --input_type encoded_image_string_tensor
    # --pipeline_config_path $SRC_DIR/faster_rcnn_resnet101_pets.config
    # --trained_checkpoint_prefix $TRAINING_DIR/$BUILD/model.ckpt-${CHECKPOINT}
    # --output_directory $TRAINING_DIR/model/$BUILD;
    # python -c "from mlboardclient.api import client; m = client.Client(); m.model_upload('$MODEL_NAME', '$MODEL_VERSION', '$TRAINING_DIR/model/$BUILD/saved_model'); m.update_task_info({'model': '#/$WORKSPACE_NAME/catalog/mlmodel/$MODEL_NAME/versions/$MODEL_VERSION'})"


if __name__ == '__main__':
    main()
