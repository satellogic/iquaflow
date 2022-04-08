# from __future__ import annotations

import glob
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ds_exceptions import DSAnnotationsNotFound, DSNotFound

DUPL_ESC_KEY = "do not duplicate dataset"


class DSWrapper:
    """
    Object used for containing dataset metainformation. Given the path of a dataset automatically finds coco format json annotations, geojson annotations, csv annotations
    and the folder that contains the images.
    Besides also provides a summary dictionary of the metadata of the dataset.

    Args:
        data_path: str. Root path of the dataset
        parent_folder:  str. Path of the folder containing the dataset
        data_input: str. Path of the folder that contains the images
        json_annotations: str. Path to the jsn COCO annotations
        params: dict. Contains metainfomation of the dataset

    Attributes:
        data_path: str. Root path of the dataset
        parent_folder:  str. Path of the folder containing the dataset
        data_input: str. Path of the folder that contains the images
        json_annotations: str. Path to the jsn COCO annotations
        params: dict. Contains metainfomation of the dataset
    """

    def __init__(
        self,
        data_path: str = "",
        data_input: str = "",
        json_annotations: str = "",
        geojson_annotations: str = "",
        mask_annotations_dir: str = "",
        ds_modifier: Optional["DSModifier"] = None,
    ):
        self.mask_annotations_dir = mask_annotations_dir

        # Initialization if data_path is provided
        if not data_path == "":
            if not os.path.exists(data_path):
                raise DSNotFound()

            self.data_path = data_path
            self.parent_folder = os.path.dirname(self.data_path)
            # Image folder
            if data_input == "":
                data_input_list = [
                    os.path.join(self.data_path, o)
                    for o in os.listdir(self.data_path)
                    if os.path.isdir(os.path.join(self.data_path, o))
                    and not o.endswith("_mask")
                    and not ("stats" in o)
                ]
                if len(data_input_list):
                    self.data_input = data_input_list[0]
                else:
                    self.data_input = ""
                    print(
                        'Warning: Could not find data_input images folder. Check that the images subfolder does not contain "stats" or "_mask"'
                    )
            else:
                self.data_input = data_input

            # raster mask annotation
            if self.data_input:
                if mask_annotations_dir == "" and os.path.exists(
                    self.data_input + "_mask"
                ):
                    self.mask_annotations_dir = self.data_input + "_mask"

            # COCO annotation
            if json_annotations == "":
                json_annotations_lists = glob.glob(
                    os.path.join(self.data_path, "*.json")
                )
                self.json_annotations = (
                    json_annotations_lists[0] if len(json_annotations_lists) else None
                )
            else:
                self.json_annotations = json_annotations
            # GeoJson annotation
            if geojson_annotations == "":
                geojson_annotations_lists = glob.glob(
                    os.path.join(self.data_path, "*.geojson")
                )
                self.geojson_annotations = (
                    geojson_annotations_lists[0]
                    if len(geojson_annotations_lists)
                    else None
                )
            else:
                self.geo_json_annotations = geojson_annotations
            # if len(self.get_annotations()) == 0:
            #    raise DSNotFound() #do not raise error if there are no annotations

        else:
            # Initialization if data_path is NOT provided
            assert (not data_input == "" and not json_annotations == "") or (
                not data_input == "" and not geojson_annotations == ""
            )
            self.geojson_annotations = (
                geojson_annotations if geojson_annotations != "" else None
            )
            self.json_annotations = json_annotations if json_annotations != "" else None
            if not os.path.exists(json_annotations) and not os.path.exists(
                geojson_annotations
            ):
                raise DSAnnotationsNotFound()
            self.data_input = data_input
            if not os.path.exists(data_input):
                raise DSNotFound()
            self.data_path = os.path.dirname(self.data_input)
            self.parent_folder = os.path.dirname(self.data_path)

        self.ds_name = os.path.basename(self.data_path)
        self.ds_modifier_applied = ds_modifier
        self.params = {"ds_name": self.ds_name}
        if self.ds_modifier_applied:
            modifier_params = self.ds_modifier_applied.log_parameters()
            self.set_log_parameters(modifier_params)

    def log_parameters(self) -> Dict[str, Any]:
        """Metainfomration logs
        Returns a dictionary containg metainformation that is intended to be logged

        Returns:
            A dict with dataset metainformation
        """
        return self.params.copy()

    def set_log_parameters(self, update_dict: Dict[str, Any]) -> None:
        """Update metainformation logs
        Allows to extend the metaparemer dictionary

        Args:
        update_dict: dict. Update metainformation

        Returns:
            A dict with dataset metainformation
        """
        self.params.update(update_dict)

    def modify(self, ds_modifier: "DSModifier") -> "DSWrapper":
        """Modify Dataset
        Returns a new DSWrapper modified by a DSModifier
        Args:
            ds_modifer: DSModifier. Object to modify the current DSWrapper
        Returns:
            A new DSWrapper modified by DSModifier
        """
        new_ds_wrapper = ds_modifier.modify_ds_wrapper(self)
        return new_ds_wrapper

    def get_annotations(self) -> List[str]:
        ann = []
        if self.json_annotations:
            ann.append(self.json_annotations)
        if self.geojson_annotations:
            ann.append(self.geojson_annotations)
        return ann


class DSModifier:
    """
    Base class that modifies a datasets. It can be used santd alone by passing the dataset path or by passing a DSWrapper.
    It can be composed by other DSModifier, inorder to create a chain of dataset modification. Besides the modification functionality, it povides a dictionary
    with metainformation of th emodificator.

    Args:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """

    def __init__(self, ds_modifier: Optional["DSModifier"] = None):
        self.name = "base_modifier"
        self.ds_modifier = ds_modifier
        self.params = {"modifier": "{}".format(self._get_name())}
        self._symlink = False

    def _get_name(self) -> str:
        """Complete name
        Provides the complete name of composed modifers

        Returns: st. Complete name of the modifer
        """
        if self.ds_modifier is None:
            return self.name
        return "{}#{}".format(self.ds_modifier._get_name(), self.name)

    def _toggle_on_symlink_for_base_modifier(self) -> None:
        self.name = DUPL_ESC_KEY
        self._symlink = True

    def _toggle_off_symlink_for_base_modifier(self) -> None:
        self.name = "base_modifier"
        self._symlink = False

    def log_parameters(self) -> Dict[str, Any]:
        """Metainfomration logs
        Returns a dictionary containg metainformation that is intended to be logged

        Returns:
            A dict with dataset metainformation
        """
        return self.params.copy()

    def set_log_parameters(self, update_dict: Dict[str, Any]) -> None:
        """Update metainformation logs
        Allows to extend the metaparemer dictionary

        Args:
        update_dict: dict. Update metainformation

        Returns:
            A dict with dataset metainformation
        """
        self.params.update(update_dict)

    def modify(
        self,
        data_path: str = "",
        data_input: str = "",
        ds_wrapper: Optional[DSWrapper] = None,
    ) -> Tuple[str, str, str]:
        """Modify Dataset
        Modiffies the images of a dataset. This method receives as input the dataset location, executes child modifications
        and then executes its current modifications. The modification woks by passing at leas one of the input arguments.
        The general approach is creating a sibling  modified datset to the original one.
        This original implementation only creats an exact copy of the original dataset.

        Args:
            data_path: str. Path of the root folder of the dataset
            data_input: str. Path of the folder containing images
            ds_wrapper: DSWrapper.
        Returns:
            mod_data_input: str. Path of the folder containing modified images
            mod_data_path: str. Path of the modified root folder of the dataset
            parent_folder: str. Path of the parent folder

        """
        if ds_wrapper is None:
            if data_path == "":
                data_path = os.path.dirname(data_input)
            ds_wrapper = DSWrapper(data_path=data_path)
        else:
            ds_wrapper = ds_wrapper

        if self.ds_modifier:
            mod_data_input, mod_data_path, mod_parent_folder = self.ds_modifier.modify(
                ds_wrapper=ds_wrapper
            )
            ds_wrapper = DSWrapper(data_path=mod_data_path)

        mod_data_input, mod_data_path, parent_folder = self._ds_modification(ds_wrapper)
        return mod_data_input, mod_data_path, parent_folder

    def modify_ds_wrapper(self, ds_wrapper: DSWrapper) -> "DSWrapper":
        """Modify DSWrapper
        Same method as modify, but it only recives and return a DSWrapper

        Args:
            ds_wrapper: DSWrapper.
        Returns:
            ds_wrapper: DSWrapper.
        """
        if self.name == DUPL_ESC_KEY:
            return ds_wrapper
        else:
            mod_data_input, mod_data_path, parent_folder = self.modify(
                ds_wrapper=ds_wrapper
            )
            return DSWrapper(data_path=mod_data_path, ds_modifier=self)

    def _ds_modification(self, ds_wrapper: DSWrapper) -> Tuple[str, str, str]:
        """New dataset creation
        This internal method creates the folder structure of the new modified dataste. First it create sthe new folder,
        then it copies all its annotations and the applyes modifications.

        Args:
            ds_wrapper: DSWrapper.

        Returns:
            data_input_modified: str. Path of the folder containing modified images
            mod_path: str. Path of the modified root folder of the dataset
            parent_folder: str. Path of the parent folder

        """
        data_path = ds_wrapper.data_path
        parent_folder = ds_wrapper.parent_folder
        name = os.path.basename(data_path)
        new_name = "{}#{}".format(name, self.name)
        mod_path = os.path.join(parent_folder, new_name)
        os.makedirs(mod_path, exist_ok=True)
        self._copy_annotations(ds_wrapper, mod_path)
        new_input_name = self._ds_input_modification(
            str(ds_wrapper.data_input), mod_path
        )

        data_input_modified = os.path.join(mod_path, new_input_name)
        return data_input_modified, mod_path, parent_folder

    def _copy_annotations(self, ds_wrapper: DSWrapper, destiniy_path: str) -> None:
        """Copy annotations
        This internal method copies all annotations of the original dataset.

        Args:
            ds_wrapper: DSWrapper.
        """
        for original_ann in ds_wrapper.get_annotations():
            name = os.path.basename(original_ann)
            dst = os.path.join(destiniy_path, name)
            shutil.copyfile(original_ann, dst)

        if ds_wrapper.mask_annotations_dir != "":
            src = ds_wrapper.mask_annotations_dir
            dst = os.path.join(
                destiniy_path, os.path.basename(ds_wrapper.mask_annotations_dir)
            )
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)

    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        This method should be overwitten for child classes. In this original method , the image sof the original
        dataset are copied to the new dataset

        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset

        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        if self._symlink:
            command = f"ln -s {data_input} {dst}"
            print(f"Symlink... {command}")
            os.system(command)
        else:
            try:
                shutil.copytree(data_input, dst)
            except OSError as e:
                # If the error was caused because the source wasn't a directory
                print("Directory not copied. Error: %s" % e)
        return input_name


class DSModifier_dir(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder. Base class for single-file modifiers.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """

    def __init__(self, ds_modifier: Optional[DSModifier] = None):
        self.name = "dir_modifier"
        self.ds_modifier = ds_modifier
        self.params = {"modifier": "{}".format(self._get_name())}

    # def is_input_file(self, data_file):
    #    return all(omi not in data_file for omi in self.omit) and any(ext in data_file for ext in self.extensions)

    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path

        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset

        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        for data_file in os.listdir(data_input):
            file_path = os.path.join(data_input, data_file)
            if os.path.isdir(file_path):
                continue
            loaded = cv2.imread(file_path, -1)
            assert loaded.ndim == 2 or loaded.ndim == 3, (
                "(load_img): File " + file_path + " not valid image"
            )
            # size_raw+= np.size(loaded)
            imgp = self._mod_img(loaded)
            # print(loaded.shape)
            cv2.imwrite(os.path.join(dst, data_file), imgp)
        return input_name

    def _mod_img(self, img: np.array) -> np.array:
        """Modify single image
        This method should be overwitten for child classes. In this Base version just return the image unchanged.

        Args
            img: Numpy array. Original image

        Returns:
            img: Numpy array. Modified image
        """
        return img  # just leave the img as is
