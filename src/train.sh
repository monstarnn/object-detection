#!/usr/bin/env bash

sed -i -e "s|%DATADIR%|$DATA_DIR|g" faster_rcnn_resnet101_pets.config

python -m object_detection/train --train_dir $TRAINING_DIR/$BUILD_ID --pipeline_config_path faster_rcnn_resnet101_pets.config
