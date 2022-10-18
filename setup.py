from setuptools import setup
import versioneer

setup(
    name='iquaflow',
    author='Pau Gallés, Katalin Takáts, Emilio Tylson, David Berga, Miguel Hernández-Cabronero, Luciano Pega, Laura Riordan-Chen, Clara Garcia, Guillermo Becker, David Vilaseca, Javier Marín',
    description='Image quality framework',
    long_description=open('README.md', encoding='utf-8').read(),
    version=versioneer.get_version(),
    maintainer_email='iquaflow@satellogic.com',
    license="MIT",
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/satellogic/iquaflow/',
    download_url=(
        'https://github.com/satellogic/iquaflow/tags/v' +
        versioneer.get_version()
    ),
    package_data={
        'iquaflow': [
            'quality_metrics/cfgs_example/*.cfg',
            'quality_metrics/cfgs_best/*.cfg',
            'quality_metrics/cfgs_multihead/*.cfg',
            'datasets/dataset_labels/RER/*.json',
            'datasets/dataset_labels/SNR/*.json'
        ]
    },
    include_package_data=True
)
