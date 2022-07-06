.. iquaflow documentation master file, created by
   sphinx-quickstart on Fri Oct 23 17:01:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to iquaflow's documentation!
=====================================
**iquaflow** is an image quality framework that aims at providing a set of tools to assess image quality. 

There are several aproaches to acomplish the objective. One can measure quality directly by applying metrics on the images that compose a dataset. These metrics can be  based on similarity (by comparing against a groundtruth) or they can also be "blind" ( such as measuring blur, noise, sharpness...). The user can add custom metrics that can be easily integrated in iquaflow. 

Furthermore, iquaflow allows to measure quality by using the performance of AI models trained on the images as a proxy. This also helps to easily make studies of performance degradation on several modifications of the original dataset with images (such as compression modifications). All this is wrapped in mlflow: an interactive tool used to visualize and summarize the results. A study like that is named a "use case" and there is a `cookiecutter <https://github.com/satellogic/iquaflow-use-case-cookiecutter>`_ to falicitate the code generation for each new study.


 
The `source code`_ and `issue tracker`_ are hosted on GitHub, and all contributions and feedback are more than welcome. 

.. _`source code`: https://github.com/satellogic/iquaflow
.. _`issue tracker`: https://github.com/satellogic/iquaflow/issues

Installation
------------

You can install iquaflow using pip::

  pip install -e . 

Alternatively you can use docker. The Dockerfile and a whole template for a use case study repository is available as a cookiecutter in `here <https://github.com/satellogic/iquaflow-use-case-cookiecutter>`_. 

iquaflow is a Python library, and therefore should work on Linux, OS X and Windows provided that you can install its dependencies. If you find any problem,
`please open an issue`_ and we will take care of it.

.. _`please open an issue`: https://github.com/satellogic/iquaflow/issues/new

.. warning::

    It is recommended that you **never ever use sudo** with pip because you might
    seriously break your system. Use `venv`_, `Pipenv`_, `pyenv`_ or `conda`_
    to create an isolated development environment instead.

.. _`venv`: https://docs.python.org/3/library/venv.html
.. _`Pipenv`: https://docs.pipenv.org/
.. _`pyenv`: https://github.com/pyenv/pyenv
.. _`conda`: https://conda.io/docs/


Conventions
------------
In **iquaflow** conventions are prefered over configurations. 

Dataset Formats
^^^^^^^^^^^^^^^^

-  IQToolBox understands a dataset as a folder containing a sub-folder
   with images and ground truth in json format. Datasets that does not
   follow this format should be changed in order to perform experiments.

-  In case of detection or segmentation tasks, the preferred formats
   are:

   -  Json in COCO format.
   -  GeoJson with the minimum required fields ("image_filename", "class_id", "geometry").
   -  A folder named maskes with images corresponding to the segmentation annotations.

-  IQToolbox primarily works with COCO json ground truth adopted by most
   of the datasets and models of the field. In case that the dataset is
   in other format, the user can transform it to COCO
   https://blog.roboflow.ai/how-to-convert-annotations-from-voc-xml-to-coco-json/
   Otherwise, IQToolBox can not perform sanity neither statistics checks

-  For other kind of tasks, such as image generation, it is only
   necessary to have the ground truth in a json format. Alternatively,
   IQT can recognize a dataset without any ground truth file

-  When the dataset is modified, iquaflow creates a modified copy of the
   dataset in its parent folder. As a convention, iquaflow adds to the name
   of the original dataset a “#” followed by the name of the
   modification as you can see in the following image.


Output Formats
^^^^^^^^^^^^^^^^

The packaged model could write in the output temporary folder the
following files in order to be parsed as experiment parameters and
metrics:

-  **results.json**: Json with keys as the name of parameter, values as
   a number related to the metric or an array reference to a sequence of
   values of that parameter.

::

   {
    “train_f1”: 0.83,
    “val_f1”: 0.78,
    “test_f1”: 0.79,
    “train_focal_loss”: [1.34, 1.29, 1.24, …., 0.01]
    “val_focal_loss”: [1.34, 1.29, 1.24, …., 0.01]
   }

-  **output.json** :Output of the model (this allows to avoid
   reproducing experiments in the future in case it is wanted to test a
   new metric for former experiments) in a folder named output. The
   format of this json file depends on the task of the DL model.

   -  Bounding Box Detection: **output.json** consists of a COCO format
      json, containing as many elements as detections have been made in
      the dataset. Each of these elements looks as shown below.

::

   {
   "image_id" : 85
   "iscrowd" : 0
   "bbox":
   [
   522.5372924804688
   474.1499938964844
   28.968505859375
   27.19696044921875
   ]
   "area": 2427.050960971974
   "category_id": 1
   "id": 1
   score : 0.9709288477897644
   }

-  Image generation: The json may contain the relative path to the
   generated images. Imagine the packaged model is Super Resolution
   model that generates five super resolution images. The package may
   store a folder named ``generated_sr_image`` in the output temporary
   file with this five images. Hence the **output.json** should be as
   following:

::

   {
    [
      "generated_sr_image/image_1.png",
      "generated_sr_image/image_2.png",
      "generated_sr_image/image_3.png",
      "generated_sr_image/image_4.png",
      "generated_sr_image/image_5.png",
    ]
   }


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   iquaflow
   iquaflow.datasets 
   iquaflow.experiments 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
