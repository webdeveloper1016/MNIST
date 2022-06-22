# MNIST in TensorFlow

This repository demonstrates using Paperspace Gradient to train and deploy a deep learning model to recognize handwritten characters, which is a canonical sample problem in machine learning.

We build a convolutional neural network to classify the [MNIST
dataset](http://yann.lecun.com/exdb/mnist/) using the
[tf.data](https://www.tensorflow.org/api_docs/python/tf/data),
[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator),
and
[tf.layers](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf)
APIs.

# Gradient Setup

## Single Node Training on Gradient

### Install Gradient CLI

```
pip install -U gradient
```

[Please check our documentation on how to install Gradient CLI and obtain an API Key](https://docs.paperspace.com/gradient/get-started/install-the-cli)

### Create project and get the project id

[Please check our documentation on how to create a project and get the project id](https://docs.paperspace.com/gradient/get-started/managing-projects)
Your project ID will look like `pr1234567`.

### Create and start a workflow

```
gradient workflows create --name mnist-sample --projectId pr1234567
+--------------+--------------------------------------+
| Name         | ID                                   |
+--------------+--------------------------------------+
| mnist-sample | 12345678-1234-1234-1234-1234567890ab |
+--------------+--------------------------------------+

```

Clone this repo, and change directoru into it, or copy [mnist-sample.yaml](mnist-sample.yaml) to your local machine.

Then run the workflow using the workflow ID from the create workflow command above.

```
gradient workflows run --id 12345678-1234-1234-1234-1234567890ab --path mnist-sample.yaml
```

That's it!

### Exporting a Model for inference

#### Export your Tensorflow model

In order to serve a Tensorflow model, simply export a SavedModel from your Tensorflow program. [SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md) is a language-neutral, recoverable, hermetic serialization format that enables higher-level systems and tools to produce, consume, and transform TensorFlow models.

Please refer to [Tensorflow documentation](https://www.tensorflow.org/guide/saved_model#save_and_restore_models) for detailed instructions on how to export SavedModels.

#### Example code showing how to export your model:

```
tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

#Starting to Export model
image = tf.placeholder(tf.float32, [None, 28, 28])
input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
mnist_classifier.export_savedmodel(<export directory>,
                                    input_fn,
                                    strip_default_attrs=True)
#Model Exported
```

We use TensorFlow's [SavedModelBuilder module](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py) to export the model. SavedModelBuilder saves a "snapshot" of the trained model to reliable storage so that it can be loaded later for inference.

For details on the SavedModel format, please see the documentation at [SavedModel README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).

For export directory, be sure to set it to `PS_MODEL_PATH` when running a model deployment on Gradient:

```
export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH'))
```

You can also use Gradient SDK to ensure you have the correct path:

```
from gradient_sdk.utils import data_dir, model_dir, export_dir
```

# (Optional) Local Setup using a Virtual Environment

Users sometimes run into local machine environment issues when trying to use Python. A common solution for this is to create and use a Python virtual environment to run Python from within. To do so:

1. Create and activate a Python virtual environment (we recommend using python3.7+):

```
cd mnist-sample

python3 -m venv venv

source venv/bin/activate
```

2. Install the required Python packages:

```
pip install -r requirements-local.txt
```

# Local Training

To train a the mnist model locally:

1. Make sure you have the latest version of TensorFlow installed.

2. Also make sure you've [added the models folder to your Python path](https://github.com/mlcommons/training/blob/master/image_classification/tensorflow/official/README.md#running-the-models); otherwise you may encounter an error like `ImportError: No module named mnist`.

3. Download the code from GitHub:

```
git clone git@github.com:Paperspace/mnist-sample.git
```

4. Start training the model:

```
python mnist.py
```

_Note: local training will take a long time, so be prepared to wait!_

If you want to shorten model training time, you can change the max steps parameter:

```
python mnist.py --max_steps=1500
```

The mnist dataset is downloaded to the `./data` directory.

Model results are stored in the `./models` directory.

Both directories can be safely deleted if you would like to start the training over from the beginning.

## Exporting the model to a specific directory

You can export the model into a specific directory, in the Tensorflow [SavedModel](https://www.tensorflow.org/guide/saved_model) format, by using the argument `--export_dir`:

```
python mnist.py --export_dir /tmp/mnist_saved_model
```

If no export directory is specified, the model is saved to a timestamped directory under `./models` subdirectory (e.g. `mnist-sample/models/1513630966/`).

## Testing a Tensorflow Serving-deployed model on your local machine using Docker

Open another terminal window and run the following in the directory where you cloned this repo:

```
docker run -t --rm -p 8501:8501 -v "$PWD/models:/models/mnist" -e MODEL_NAME=mnist tensorflow/serving
```

Now you can test the local inference endpoint by running:

```
python serving_rest_client_test.py
```

Optionally you can provide a path to an image file to run a prediction on:

```
python serving_rest_client_test.py --path example3.png
```

Once you've completed local testing using the tensorflow/serving docker container, stop the running container as follows:

```
docker ps
docker kill <container-id-or-name>
```

## Training the model on a node with a GPU for use with Tensorflow Serving on a node with only a CPU

If you are training on Tensorflow using a GPU but would like to export the model for use in Tensorflow Serving on a CPU-only server, you can train and/or export the model using `--data_format=channels_last`:

```
python mnist.py --data_format=channels_last
```

The SavedModel will be saved in a timestamped directory under `models` subdirectory (e.g. `mnist-sample/models/1513630966/`).

## Inspecting and getting predictions with the SavedModel file

You can also use the [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel) tool to inspect and execute the SavedModel.
