import os

from iquaflow.datasets import DSModifier_sr, DSWrapper

dataset_name = "ds_inria_dataset"
dataset_images_folder = "images"
current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current_path))  # "/iquaflow-/"
tests_path = os.path.join(root_path, "tests")
ml_models_path = os.path.join(tests_path, "test_ml_models")
mlruns_path = os.path.join(current_path, "mlruns")
base_ds = os.path.join(tests_path, "test_datasets")
ds_path = os.path.join(base_ds, dataset_name)
data_path = os.path.join(ds_path, dataset_images_folder)

experiment_name = "test_sr_algorithms"
ds_wrapper = DSWrapper(data_path=data_path)


# SRGAN case
print("Running SRGAN")
ds_modifier_SRGAN = DSModifier_sr(
    params={
        "algo": "SRGAN",
        "upscale_factor": 2,
        "arch": "srgan_2x2",
        "gpu_device": 0,
        "path_to_repo": "~/SRGAN-PyTorch",
        "path_to_model_weights": "~/SRGAN-PyTorch/weights/PSNR.pth",
    }
)
ds_modifier_SRGAN.modify(data_input=data_path)
out_dir_images = os.path.join(
    base_ds, dataset_name + "#" + ds_modifier_SRGAN._get_name(), dataset_images_folder
)
for img_path in os.listdir(out_dir_images):
    print(out_dir_images + "/" + img_path)

# CAR case
print("Running CAR")
ds_modifier_CAR = DSModifier_sr(
    params={
        "algo": "CAR",
        "upscale_factor": 2,
        "gpu_device": 0,
        "path_to_repo": "~/CAR",
        "path_to_model_weights": "~/CAR/models",
    }
)
ds_modifier_CAR.modify(data_input=data_path)
out_dir_images = os.path.join(
    base_ds, dataset_name + "#" + ds_modifier_CAR._get_name(), dataset_images_folder
)
for img_path in os.listdir(out_dir_images):
    print(out_dir_images + "/" + img_path)

# MSRN case
print("Running MSRN")
ds_modifier_MSRN = DSModifier_sr(
    params={
        "algo": "MSRN",
        "source_gsd": 0.6,
        "target_gsd": 0.3,
        "gpu_device": 0,
        "path_to_repo": "~/MSRN_EXAMPLE",
        "path_to_model_weights": "~/MSRN_EXAMPLE/model.pth",
    }
)
ds_modifier_MSRN.modify(data_input=data_path)
out_dir_images = os.path.join(
    base_ds, dataset_name + "#" + ds_modifier_MSRN._get_name(), dataset_images_folder
)
for img_path in os.listdir(out_dir_images):
    print(out_dir_images + "/" + img_path)
