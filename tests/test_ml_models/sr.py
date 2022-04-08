import argparse
import json
import os

current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_images_folder = "generated_sr_images"
base_ds = os.path.join(current_path, "test_datasets")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputpath", default=os.path.join(current_path, "tmp/"))
    parser.add_argument(
        "--trainds", default=os.path.join(base_ds, "ds_coco_dataset")
    )  # ds_coco_dataset
    args, uk_args = parser.parse_known_args()
    parser.add_argument("--valds", default=args.trainds)  # ds_coco_dataset
    parser.add_argument(
        "--traindsinput",
        default=os.path.join(args.trainds, "images"),  # test or images
    )  # default subforlder from task ds (ds_wrapper.data_input)
    args, uk_args = parser.parse_known_args()
    parser.add_argument(
        "--valdsinput",
        default=os.path.join(args.valds, "images"),  # test or images
    )
    args, uk_args = parser.parse_known_args()
    train_ds = args.trainds
    train_ds_input = args.traindsinput
    val_ds = args.valds
    val_ds_input = args.valdsinput
    output_path = args.outputpath

    # objective: read train_ds and val_ds files and apply super-resolution, save in output_path/val/generated_sr_images folder

    # alternative (old): copy input image paths as output image paths
    train_ds_output = train_ds_input
    val_ds_output = val_ds_input

    # MOCK SR: create generated_sr_images from copy
    """
    train_ds_output = os.path.join(
        output_path, "train", output_images_folder
    )  # train_ds_input
    # create subfolders
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, "train")):
        os.mkdir(os.path.join(output_path, "train"))
    if not os.path.exists(train_ds_output):
        os.mkdir(train_ds_output)
    # generate / (mock sr=copy images)
    image_files = os.listdir(train_ds_input)
    print(image_files)
    for idx, image_name in enumerate(image_files):
        if os.path.exists(train_ds_output + "/" + image_name):
            os.remove(train_ds_output + "/" + image_name)
        os.symlink(train_ds_input + "/" + image_name, train_ds_output + "/" + image_name)
    """
    """
    if val_ds != train_ds:
        val_ds_output = os.path.join(
            output_path, "val", output_images_folder
        )  # val_ds_input
        if not os.path.exists(os.path.join(output_path, "val")):
            os.mkdir(os.path.join(output_path, "val"))
        if not os.path.exists(val_ds_output):
            os.mkdir(val_ds_output)
        image_files = os.listdir(val_ds_input)
        for idx, image_name in enumerate(image_files):
            if os.path.exists(val_ds_output + "/" + image_name):
                os.remove(val_ds_output + "/" + image_name)
            os.symlink(val_ds_input + "/" + image_name, val_ds_output + "/" + image_name)
    """
    output_json = {
        "train_ds": train_ds,
        "train_ds_input": train_ds_input,
        "train_ds_output": train_ds_output,
        "val_ds": val_ds,
        "val_ds_input": val_ds_input,
        "val_ds_output": val_ds_output,
    }

    with open(os.path.join(output_path, "output.json"), "w") as f:
        json.dump(output_json, f)
