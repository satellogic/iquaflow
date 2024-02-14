# ![iquaflow](https://github.com/satellogic/iquaflow/raw/main/docs/source/iquaflow_logo_transp.png) <br /> An image quality framework

iquaflow is an image quality framework that aims at providing a set of tools to assess image quality. One of the main contributions of this framework is that it allows to measure quality by using the performance of AI models trained on the images as a proxy. The framework includes ready-to-use metrics such as SNR, MTF, FWHM or RER. It also includes modifiers to alter images (noise, blur, jpeg compression, quantization, etc). In both cases, metrics and modifiers, it is easy to implement new ones. Additionally, we include dataset preparation, sanity check and all other necessary tools to carry out new experiments. 

Usage examples and a detailed description of our framework can be found within [our documentation](http://iquaflow.readthedocs.io/) on Read the Docs.

## Use cases

[Cookiecutter use case](https://github.com/satellogic/iquaflow-use-case-cookiecutter)

[Mnist use case](https://github.com/satellogic/iquaflow-mnist-use-case)

[Single image super-resolution use case](https://github.com/satellogic/iquaflow-sisr-use-case)

[Multi-frame super-resolution use case](https://github.com/satellogic/iquaflow-mfsr-use-case)

[Oriented-object detection with compression use case](https://github.com/satellogic/iquaflow-dota-obb-use-case)

[Object detection with compression use case](https://github.com/satellogic/iquaflow-dota-use-case)

[Airplane detection use case](https://github.com/satellogic/iquaflow-airport-use-case)

[(QMRNet) Metric Regression on EO datasets use case](https://github.com/dberga/iquaflow-qmr-eo)

[(QMRNet) Metric Regression for single image super-resolution use case](https://github.com/dberga/iquaflow-qmr-sisr)

[(QMRNet) Metric Regression for super-resolution optimization use case](https://github.com/dberga/iquaflow-qmr-loss)

## Installation

You can install iquaflow using pip:

```
pip install iquaflow 
```

Read more complete installation instructions at [our documentation](http://iquaflow.readthedocs.io/).

iquaflow is a pure Python library, and therefore should work on Linux, OS X and Windows
provided that you can install its dependencies. If you find any problem,
[please open an issue](https://github.com/satellogic/iquaflow/issues/new)
and we will take care of it.

## Citation 

If you use this library in your research, please consider citing:

```
@article{Iquaflow,
  author={Gallés, Pau and Takáts, Katalin and Hernández-Cabronero, Miguel and Berga, David and Pega, Luciano and Riordan-Chen, Laura and Garcia, Clara and Becker, Guillermo and Garriga, Adan and Bukva, Anica and Serra-Sagristà, Joan and Vilaseca, David and Marín, Javier},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A New Framework for Evaluating Image Quality Including Deep Learning Task Performances as a Proxy}, 
  year={2024},
  volume={17},
  number={},
  pages={3285-3296},
  doi={10.1109/JSTARS.2023.3342475}}
```

## Support

For any questions or suggestions you can use the issues section or reach us at iquaflow@satellogic.com.
