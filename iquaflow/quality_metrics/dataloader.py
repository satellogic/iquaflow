import os
import shutil
from typing import Any, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from iquaflow.datasets import DSWrapper
from iquaflow.quality_metrics.tools import (
    check_if_contains_edges,
    check_if_contains_homogenous,
    force_rgb,
    generate_crop_permut,
    replace_crop_permut,
    split_list,
)


class Dataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(
        self,
        split_name: Any,
        data_path: Any,
        data_input: Any,
        num_crops: int = 50,
        crop_size: List[int] = [256, 256],
        split_percent: float = 1.0,
        img_size: Tuple[int, int] = (5000, 5000),
    ):
        self.split_name = split_name
        self.data_path = data_path
        self.data_input = data_input
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.split_percent = split_percent
        self.img_size = img_size
        # list images to split
        lists_files = [
            self.data_input + "/" + filename for filename in os.listdir(self.data_input)
        ]
        self.lists_files = split_list(lists_files, split_percent)
        self.lists_mod_files: List[Any] = []
        self.lists_crop_files: List[Any] = []
        # keys
        self.mod_keys: List[str] = []
        self.crop_mod_keys: List[str] = []
        # params
        self.mod_params: List[Any] = []
        self.crop_mod_params: List[Any] = []
        self.mod_resol: List[Any] = []
        # transforms
        self.tCROP = transforms.Compose(
            [
                transforms.RandomCrop(size=(self.crop_size[0], self.crop_size[1])),
            ]
        )
        self.cCROP = transforms.Compose(
            [
                transforms.CenterCrop(size=(self.crop_size[0], self.crop_size[1])),
            ]
        )

    def __len__(self) -> int:
        """
        if len(self.lists_mod_files) is 0:
            return len(self.lists_mod_files)*self.num_crops
        else:
            return len(self.lists_files)*self.num_crops*len(self.lists_mod_files)
        """
        if len(self.lists_crop_files) > 0:
            return len(self.lists_crop_files)
        elif len(self.lists_mod_files) > 0:
            return len(self.lists_mod_files) * len(self.lists_files)
        else:
            return len(self.lists_files)

    def __getitem__(self, idx):
        if (
            len(self.lists_crop_files) > 0 and len(self.lists_mod_files) > 0
        ):  # cropped and modified
            filename = self.lists_crop_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image)  # .unsqueeze_(0)
            x = force_rgb(image_tensor)  # usgs case (nth dimensions, e.g. depth)
            y = torch.tensor(self.crop_mod_params[idx])
            param = self.crop_mod_keys[idx]
        elif (
            len(self.lists_crop_files) == 0 and len(self.lists_mod_files) != 0
        ):  # modified but not cropped
            filename = self.lists_mod_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image)  # .unsqueeze_(0)
            x = force_rgb(image_tensor)  # usgs case (nth dimensions, e.g. depth)
            y = torch.tensor(self.mod_params[idx])
            param = self.mod_keys[idx]
        elif (
            len(self.lists_crop_files) > 0 and len(self.lists_mod_files) == 0
        ):  # cropped but no modified
            filename = self.lists_crop_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image)  # .unsqueeze_(0)
            x = force_rgb(image_tensor)  # usgs case (nth dimensions, e.g. depth)
            y = torch.tensor(0)
            param = ""
        else:
            filename = self.lists_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image)  # .unsqueeze_(0)
            x = force_rgb(image_tensor)
            y = torch.tensor(0)
            param = ""

        return filename, param, Variable(x), Variable(y)

    def __modify__(self, ds_modifiers: Any, overwrite: Any = False) -> Any:
        self.lists_mod_files = []  # one per each modifier
        self.mod_keys = []  # one per each modifier
        self.mod_params = []  # one per each modifier
        for midx in range(len(ds_modifiers)):
            ds_modifier = ds_modifiers[midx]
            mod_key = next(
                iter(ds_modifier.params.keys())
            )  # parameter name of modifier
            mod_param = next(
                iter(ds_modifier.params.values())
            )  # parameter value of modifier
            print(
                "Preprocessing dataset with modifiers: "
                + mod_key
                + " "
                + str(mod_param)
            )
            ds_wrapper = DSWrapper(data_path=self.data_path, data_input=self.data_input)
            # ds_wrapper.data_input=self.data_input
            output_dir = self.data_path + "#" + ds_modifier.name
            split_dir = os.path.join(output_dir, self.split_name)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            if not os.path.exists(split_dir):
                os.mkdir(split_dir)
            elif overwrite is True:  # remove (overwrite)
                shutil.rmtree(split_dir)
                os.mkdir(split_dir)

            # reuse if existing modified images (same exact ones)
            basenames_lists_files = [
                os.path.basename(file) for file in self.lists_files
            ]
            list_mod_same_files = [
                os.path.join(split_dir, file)
                for file in os.listdir(split_dir)
                if file in basenames_lists_files
            ]
            """ # (alternative) reuse whole folder
            if len(os.listdir(split_dir)) >= len(self.lists_files):
                list_mod_files = [
                    split_dir + "/" + filename for filename in os.listdir(split_dir)
                ]
            """
            if len(list_mod_same_files) == len(self.lists_files):
                list_mod_files = list_mod_same_files
            else:
                ds_wrapper_modified = ds_modifier.modify_ds_wrapper(
                    ds_wrapper=ds_wrapper
                )
                if len(os.listdir(ds_wrapper_modified.data_input)) == 0:
                    old_dir = os.path.join(
                        ds_wrapper_modified.data_path,
                        os.path.basename(ds_wrapper.data_input),
                    )
                else:
                    old_dir = ds_wrapper_modified.data_input

                for file_name in os.listdir(old_dir):
                    shutil.move(
                        os.path.join(old_dir, file_name),
                        os.path.join(split_dir, file_name),
                    )
                shutil.rmtree(old_dir)
                ds_wrapper_modified.data_input = split_dir
                list_mod_files = [
                    ds_wrapper_modified.data_input + "/" + filename
                    for filename in os.listdir(ds_wrapper_modified.data_input)
                ]
            list_mod_files = split_list(list_mod_files, self.split_percent)
            self.lists_mod_files.append(list_mod_files)
            self.mod_keys.append(mod_key)
            self.mod_params.append(mod_param)
            # print(self.lists_mod_files[-1])
            self.img_size = Image.open(self.lists_mod_files[midx][0]).size

    def __crop__(self, overwrite: bool = False) -> None:
        self.list_crop_files: List[str] = []
        self.crop_mod_keys = []  # one per each modifier
        self.crop_mod_params = []  # one per each modifier
        self.mod_resol = []  # one per each modifier
        if len(self.lists_mod_files) != 0:  # generate crops from modifiers
            # generating crops permutation
            num_images = len(self.lists_mod_files[0])
            self.crops_permut_y, self.crops_permut_x = generate_crop_permut(
                self.num_crops, num_images, self.img_size, self.crop_size
            )
            # for each modifier
            for midx, list_mod_files in enumerate(self.lists_mod_files):
                # check crops folder
                filename_ini = self.lists_mod_files[midx][0]
                crops_folder = (
                    os.path.dirname(filename_ini)
                    + "_"
                    + str(self.num_crops)
                    + "crops"
                    + str(self.crop_size[0])
                    + "x"
                    + str(self.crop_size[1])
                )
                if not os.path.exists(crops_folder):
                    os.mkdir(crops_folder)
                elif overwrite is True:
                    shutil.rmtree(crops_folder)
                    os.mkdir(crops_folder)
                # for each sample
                for idx, mod_files in enumerate(list_mod_files):
                    filename = self.lists_mod_files[midx][idx]
                    filename_noext = os.path.splitext(os.path.basename(filename))[0]
                    # cropping
                    for cidx in range(self.num_crops):
                        filename_cropped = (
                            crops_folder
                            + "/"
                            + filename_noext
                            + "_crop"
                            + str(cidx + 1)
                            + ".png"
                        )
                        if not os.path.exists(filename_cropped) or overwrite is True:
                            print(
                                "Generating crop ("
                                + str(cidx + 1)
                                + "/"
                                + str(self.num_crops)
                                + ")"
                                + " for "
                                + filename_noext
                                + " with "
                                + str(self.mod_keys[midx])
                                + " "
                                + str(self.mod_params[midx])
                            )
                            image = Image.open(filename)
                            image_tensor = transforms.functional.to_tensor(
                                image
                            ).unsqueeze_(0)
                            # else: reuse image if existing
                            # preproc_image = self.tCROP(image_tensor)
                            preproc_image = transforms.functional.crop(
                                image_tensor,
                                self.crops_permut_y[cidx][idx],
                                self.crops_permut_x[cidx][idx],
                                self.crop_size[0],
                                self.crop_size[1],
                            )
                            crop_array = np.array(
                                transforms.functional.to_pil_image(
                                    (torch.squeeze(preproc_image))
                                )
                            )
                            if self.mod_keys[midx] == "rer":
                                # check crop requirement 100 times (at max)
                                check_count = 100
                                for _ in range(check_count):
                                    check = check_if_contains_edges(
                                        crop_array, 0, 100, 0.002
                                    )
                                    if check is True:  # break loop if check is true
                                        break
                                    (
                                        self.crops_permut_y[cidx][idx],
                                        self.crops_permut_x[cidx][idx],
                                    ) = replace_crop_permut(
                                        self.crops_permut_y[cidx][idx],
                                        self.crops_permut_x[cidx][idx],
                                        1,
                                        self.img_size,
                                        self.crop_size,
                                    )
                                    preproc_image = transforms.functional.crop(
                                        image_tensor,
                                        self.crops_permut_y[cidx][idx],
                                        self.crops_permut_x[cidx][idx],
                                        self.crop_size[0],
                                        self.crop_size[1],
                                    )
                                    crop_array = np.array(
                                        transforms.functional.to_pil_image(
                                            (torch.squeeze(preproc_image))
                                        )
                                    )
                            elif self.mod_keys[midx] == "snr":
                                # check crop requirement 100 times (at max)
                                check_count = 100
                                for _ in range(check_count):
                                    check = check_if_contains_homogenous(
                                        crop_array, 0, 30, 0.35
                                    )
                                    if check is True:  # break loop if check is true
                                        break
                                    (
                                        self.crops_permut_y[cidx][idx],
                                        self.crops_permut_x[cidx][idx],
                                    ) = replace_crop_permut(
                                        self.crops_permut_y[cidx][idx],
                                        self.crops_permut_x[cidx][idx],
                                        1,
                                        self.img_size,
                                        self.crop_size,
                                    )
                                    preproc_image = transforms.functional.crop(
                                        image_tensor,
                                        self.crops_permut_y[cidx][idx],
                                        self.crops_permut_x[cidx][idx],
                                        self.crop_size[0],
                                        self.crop_size[1],
                                    )
                                    crop_array = np.array(
                                        transforms.functional.to_pil_image(
                                            (torch.squeeze(preproc_image))
                                        )
                                    )
                            self.mod_resol.append(image.size)
                            save_image(preproc_image, filename_cropped)
                        else:
                            print(
                                f"{os.path.basename(filename_cropped)} already exists"
                            )
                        self.lists_crop_files.append(filename_cropped)
                        self.crop_mod_keys.append(self.mod_keys[midx])
                        self.crop_mod_params.append(self.mod_params[midx])
                        # print(self.lists_crop_files[-1])  # print last sample name
                    # os.remove(filename) # remove modded image to clean disk
        else:  # generate crops from real images
            # generating crops permutation
            num_images = len(self.lists_files)
            self.crops_permut_y, self.crops_permut_x = generate_crop_permut(
                self.num_crops, num_images, self.img_size, self.crop_size
            )
            # check crops folder
            filename_ini = self.lists_files[0]
            crops_folder = (
                os.path.dirname(filename)
                + "_"
                + str(self.num_crops)
                + "crops"
                + str(self.crop_size[0])
                + "x"
                + str(self.crop_size[1])
            )
            if not os.path.exists(crops_folder):
                os.mkdir(crops_folder)
            elif overwrite is True:
                shutil.rmtree(crops_folder)
                os.mkdir(crops_folder)
            # for each sample
            for idx, file in enumerate(self.lists_files):
                filename = self.lists_files[idx]
                filename_noext = os.path.splitext(os.path.basename(filename))[0]
                # cropping
                for cidx in range(self.num_crops):
                    filename_cropped = (
                        crops_folder
                        + "/"
                        + filename_noext
                        + "_crop"
                        + str(cidx + 1)
                        + ".png"
                    )
                    if not os.path.exists(filename_cropped) or overwrite is True:
                        print(
                            "Generating crop ("
                            + str(cidx + 1)
                            + "/"
                            + str(self.num_crops)
                            + ")"
                            + " for "
                            + filename_noext
                        )
                        image = Image.open(filename)
                        image_tensor = transforms.functional.to_tensor(
                            image
                        ).unsqueeze_(0)
                        # preproc_image = self.tCROP(image_tensor)
                        preproc_image = transforms.functional.crop(
                            image_tensor,
                            self.crops_permut_y[cidx][idx],
                            self.crops_permut_x[cidx][idx],
                            self.crop_size[0],
                            self.crop_size[1],
                        )
                        save_image(preproc_image, filename_cropped)
                        self.mod_resol.append(image.size)
                    else:
                        print(f"{os.path.basename(filename_cropped)} already exists")
                    self.lists_crop_files.append(filename_cropped)
                    # print(self.lists_crop_files[-1])  # print last sample name
                # os.remove(filename)  # remove image to clean disk
