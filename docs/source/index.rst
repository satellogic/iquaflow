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


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   UserGuide.ipynb
   Course.ipynb
   iquaflow


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
