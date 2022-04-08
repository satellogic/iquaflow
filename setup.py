from setuptools import setup
import versioneer

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    package_data={
        'iquaflow': [
            'quality_metrics/cfgs_example/*.cfg',
            'datasets/dataset_labels/RER/*.json',
            'datasets/dataset_labels/SNR/*.json'
        ]
    },
    include_package_data=True
)
