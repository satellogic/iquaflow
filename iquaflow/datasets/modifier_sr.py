import os
import shutil
from typing import Any, Dict, Optional

from PIL import Image

from iquaflow.datasets import DSModifier


class DSModifier_sr(DSModifier):
    """
    Class that modifies a dataset in a folder compressing it with JPG encoding at given quality.

    Args:
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier (at least 'quality')

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """

    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo": "MSRN",
            "source_gsd": 0.6,
            "target_gsd": 0.3,
            "gpu_device": 0,
            "path_to_repo": "",
            "path_to_model_weights": "",
        },
    ):
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier

        if "source_gsd" not in self.params.keys() and (
            "upscale_factor" not in self.params.keys()
            or "target_gsd" not in self.params.keys()
        ):  # no param provided (set default origin gsd)
            self.params["source_gsd"] = 0.6
        if (
            "upscale_factor" not in self.params.keys()
        ):  # calc rounded upscale factor if sgsd and tgsd provided
            self.params["upscale_factor"] = int(
                self.params["source_gsd"] / self.params["target_gsd"]
            )  # gsd to scale factor
        if (
            "target_gsd" not in self.params.keys()
        ):  # calc tgsd if sgsd and upscale_factor provided
            self.params["target_gsd"] = self.params["source_gsd"] / float(
                self.params["upscale_factor"]
            )
        if (
            "source_gsd" not in self.params.keys()
        ):  # calc sgsd if upscale_factor and tgsd provided
            self.params["source_gsd"] = self.params["target_gsd"] * float(
                self.params["upscale_factor"]
            )

        self.name = f"{params['algo']}_{params['source_gsd']}_to_{params['target_gsd']}_modifier"
        self.params.update({"modifier": "{}".format(self._get_name())})

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
            # print("Processing ", data_file)
            file_path = os.path.join(data_input, data_file)
            # loaded = cv2.imread(file_path, -1)
            # assert loaded.ndim == 2 or loaded.ndim == 3, (
            #    "(load_img): File " + file_path + " not valid image"
            # )
            # size_raw+= np.size(loaded)
            dst_file = os.path.join(dst, data_file)
            imgp = self._mod_img(file_path, dst)
            # print(loaded.shape)
            # cv2.imwrite(os.path.join(dst, data_file), imgp)
            if imgp != dst_file:
                if (
                    os.path.splitext(os.path.basename(imgp))[1]
                    != os.path.splitext(os.path.basename(dst_file))[1]
                ):
                    Image.open(imgp).save(dst_file)  # rewrite in new format
                    os.remove(imgp)  # clean
                else:
                    shutil.move(imgp, dst_file)  # move file location / rename
        return input_name

    def _mod_img(self, filename: str, path_out: str) -> str:
        name = os.path.basename(filename).split(".")[0]
        gpu_device = self.params["gpu_device"]
        path_to_repo = os.path.abspath(os.path.expanduser(self.params["path_to_repo"]))
        path_to_model_weights = os.path.abspath(
            os.path.expanduser(self.params["path_to_model_weights"])
        )
        sgsd = self.params["source_gsd"]
        tgsd = self.params["target_gsd"]
        upscale_factor = self.params["upscale_factor"]
        if self.params["algo"] == "MSRN":
            if path_to_repo == "":
                path_to_repo = "MSRN_EXAMPLE"
            if path_to_model_weights == "":
                path_to_model_weights = path_to_repo + "/model.pth"
            path_to_algorithm = path_to_repo + "/main.py"
            python_script_format = "python3 {} --filename {} --gpu_device {} --source_res {} --target_res {} --path_to_model_weights {} --path_out {}"
            script = python_script_format.format(
                path_to_algorithm,
                filename,
                gpu_device,
                sgsd,
                tgsd,
                path_to_model_weights,
                path_out,
            )
            os.system(script)
            out_file_path = os.path.join(
                path_out,
                name + "_sr_" + str.replace(str(tgsd), ".", "") + "m" + "." + "png",
            )
        if self.params["algo"] == "SRGAN":
            arch = self.params["arch"]
            fmt_out = os.path.splitext(filename)[1][1::]
            if path_to_repo == "":
                path_to_repo = "SRGAN-PyTorch"
            if path_to_model_weights == "":
                path_to_model_weights = path_to_repo + "/weights/PSNR.pth"
            path_to_algorithm = path_to_repo + "/test_image.py"
            # upscale_factor = self.params["upscale-factor"]
            python_script_format = "python3 {} --lr {} --gpu {} --upscale-factor {} --model-path {} -a {} --path_out {}"
            script = python_script_format.format(
                path_to_algorithm,
                filename,
                gpu_device,
                upscale_factor,
                path_to_model_weights,
                arch,
                path_out,
            )
            os.system(script)  # debug this
            out_file_path = os.path.join(path_out, "sr_" + name + "." + fmt_out)
            tmp_bicubic_path = os.path.join(path_out, "bicubic_" + name + "." + fmt_out)
            tmp_compare_path = os.path.join(path_out, "compare_" + name + "." + fmt_out)
            tmp_lr_path = os.path.join(path_out, "lr_" + name + "." + fmt_out)
            os.remove(tmp_bicubic_path)  # clean
            os.remove(tmp_compare_path)  # clean
            os.remove(tmp_lr_path)  # clean

        if self.params["algo"] == "CAR":
            if path_to_repo == "":
                path_to_repo = "CAR"
            if path_to_model_weights == "":
                path_to_model_weights = path_to_repo + "/models"
            path_to_algorithm = path_to_repo + "/run.py"
            tmp_path_to_input = path_to_repo + "/examples"
            tmp_path_to_input_image = (
                tmp_path_to_input + "/" + os.path.basename(filename)
            )
            os.makedirs(tmp_path_to_input, exist_ok=True)
            shutil.copyfile(filename, tmp_path_to_input_image)
            python_script_format = "python3 {} --img_dir {} --scale {} --model_dir {} --output_dir {} --upscale_only"
            script = python_script_format.format(
                path_to_algorithm,
                tmp_path_to_input,
                upscale_factor,
                path_to_model_weights,
                path_out,
            )
            os.system(script)
            out_file_path = os.path.join(path_out, name + "_recon" + "." + "png")
            tmp_down_path = os.path.join(path_out, name + "_down" + "." + "png")
            tmp_orig_path = os.path.join(path_out, name + "_orig" + "." + "png")
            os.remove(tmp_down_path)  # clean
            os.remove(tmp_orig_path)  # clean
            os.remove(tmp_path_to_input_image)  # clean
        # rec_img = cv2.imread(out_file_path, -1)
        return out_file_path
