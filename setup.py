from setuptools import setup
import versioneer

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
