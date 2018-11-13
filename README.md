# Object detection

Tensorflow object-detection training

## Running training

[Object detection pets dataset](https://cloud.kuberlab.io/kuberlab-demo/catalog/dataset/object-detection-pets/readme/) contains:

 - pets tensorflow record
 - pets label map
 - pretrained coco model (downloaded from [here](http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz))

To perform training, install this project on cluster using KuberLab platform.
Edit the source named **data** and connect it to dataset **object-detection-pets** (from the link above).
Then, it is ready to start training: run task named **train**.
Training with current settings will take several hours. However, while training is running we can start
task **eval**: it takes last tensorflow training checkpoint and log some images with detections to
**tensorboard**. As the model training progresses, task **eval** can be performed many times to see the
detection correctness for the model.

## Export model

To export model, need to adjust some parameters for task **export**:

Change the execution command as follows:

* Specify **CHECKPOINT** var according to num steps in task train
* Specify **BUILD** var according to build id (task number) of task train

Then run task **export**. It will export TensorFlow saved model to the specified directory.

For saving the model to the catalog, find the job associated with task **export**,
click **options (...)** -> **Export**. Type name of the model - **object-detection-pets** (or give another name)
and version - **1.0.0** (or another).

# Run serving, request and detection

There is a pre-trained [object-detection-pets model](https://cloud.kuberlab.io/kuberlab-demo/catalog/mlmodel/object-detection-pets)
which can be used for serving already. Use the following command to start serving on KuberLab (by clicking *Serve* in model view):

```
kuberlab-serving --driver tensorflow --port=9000 --model-path=/model
```

Then open your project's jupyter and find notebook **run_serving.ipynb** in **src** directory.
Adjust here serving host to form **<serving-name>.<namespace>.svc.cluster.local**. Upload some example pet image,
say, from [here](http://www.dogexpress.in/wp-content/uploads/2017/10/Pet-Dog-Registration-Panchkula-660x330.jpg) and put
it to **pet.jpg** (or anywhere else and adjust **IMAGE_FILE**)

# Preparing Inputs

If you want to create you own dataset, please refer to section

Tensorflow Object Detection API reads data using the TFRecord file format. Two
sample scripts (`create_pascal_tf_record.py` and `create_pet_tf_record.py`) are
provided to convert from the PASCAL VOC dataset and Oxford-IIIT Pet dataset to
TFRecords.

## Generating the PASCAL VOC TFRecord files.

The raw 2012 PASCAL VOC data set is located
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
To download, extract and convert it to TFRecords, run the following commands
below:

```bash
# From src
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
python -m object_detection/dataset_tools/create_pascal_tf_record \
    --label_map_path=data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record
```

You should end up with two TFRecord files named `pascal_train.record` and
`pascal_val.record` in the `src` directory.

The label map for the PASCAL VOC data set can be found at
`data/pascal_label_map.pbtxt`.

## Generating the Oxford-IIIT Pet TFRecord files.

The Oxford-IIIT Pet data set is located
[here](http://www.robots.ox.ac.uk/~vgg/data/pets/). To download, extract and
convert it to TFRecrods, run the following commands below:

```bash
# From src
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvf annotations.tar.gz
tar -xvf images.tar.gz
python -m object_detection/dataset_tools/create_pet_tf_record \
    --label_map_path=data/pet_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=`pwd`
```

You should end up with two TFRecord files named `pet_train.record` and
`pet_val.record` in the `src` directory.

The label map for the Pet dataset can be found at
`data/pet_label_map.pbtxt`.


# Bringing your own dataset

To use your own dataset in Tensorflow Object Detection API, you must convert it
into the [TFRecord file format](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details).
This document outlines how to write a script to generate the TFRecord file.

## Label Maps

Each dataset is required to have a label map associated with it. This label map
defines a mapping from string class names to integer class Ids. The label map
should be a `StringIntLabelMap` text protobuf. Sample label maps can be found in
object_detection/data. Label maps should always start from id 1.

## Dataset Requirements

For every example in your dataset, you should have the following information:

1. An RGB image for the dataset encoded as jpeg or png.
2. A list of bounding boxes for the image. Each bounding box should contain:
    1. A bounding box coordinates (with origin in top left corner) defined by 4
       floating point numbers [ymin, xmin, ymax, xmax]. Note that we store the
       _normalized_ coordinates (x / width, y / height) in the TFRecord dataset.
    2. The class of the object in the bounding box.

# Example Image

Consider the following image:

![Example Image](img/example_cat.jpg "Example Image")

with the following label map:

```
item {
  id: 1
  name: 'Cat'
}


item {
  id: 2
  name: 'Dog'
}
```

We can generate a tf.Example proto for this image using the following code:

```python

def create_cat_tf_example(encoded_cat_image_data):
   """Creates a tf.Example proto from sample cat image.

  Args:
    encoded_cat_image_data: The jpg encoded data of the cat image.

  Returns:
    example: The created tf.Example.
  """

  height = 1032.0
  width = 1200.0
  filename = 'example_cat.jpg'
  image_format = b'jpg'

  xmins = [322.0 / 1200.0]
  xmaxs = [1062.0 / 1200.0]
  ymins = [174.0 / 1032.0]
  ymaxs = [761.0 / 1032.0]
  classes_text = ['Cat']
  classes = [1]

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example
```

## Conversion Script Outline

A typical conversion script will look like the following:

```python

import tensorflow as tf

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO: Populate the following variables from your example.
  height = None # Image height
  width = None # Image width
  filename = None # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO: Write code to read in your dataset to examples variable

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()

```

Note: You may notice additional fields in some other datasets. They are
currently unused by the API and are optional.

Note: Please refer to the section on [Running an Instance Segmentation
Model](instance_segmentation.md) for instructions on how to configure a model
that predicts masks in addition to object bounding boxes.
