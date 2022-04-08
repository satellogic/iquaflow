import os
import shutil

from iquaflow.datasets import (
    DSModifier_blur,
    DSModifier_dir,
    DSModifier_gsd,
    DSModifier_jpg,
    DSModifier_quant,
    DSModifier_rer,
    DSModifier_sharpness,
    DSModifier_snr,
)

current_path = os.path.dirname(os.path.realpath(__file__))
ds_path = os.path.join(current_path, "test_datasets", "ds_coco_dataset")


class TestModifiers:
    def test_run(self):
        img_path = os.path.join(ds_path, "images")

        # DSModifier_dir
        shutil.rmtree(ds_path + "#dir_modifier", ignore_errors=True)
        mod = DSModifier_dir()
        mod.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + "#dir_modifier/images/000000005802.jpg"
        ), "DSModifier_dir Failed"
        shutil.rmtree(ds_path + "#dir_modifier", ignore_errors=True)

        # DSModifier_jpg
        shutil.rmtree(ds_path + "#jpg85_modifier", ignore_errors=True)
        jpg85 = DSModifier_jpg(params={"quality": 85})
        jpg85.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + "#jpg85_modifier/images/000000005802.jpg"
        ), "DSModifier_jpg Failed"
        shutil.rmtree(ds_path + "#jpg85_modifier", ignore_errors=True)

        # DSModifier_quant
        shutil.rmtree(ds_path + "#quant5_modifier", ignore_errors=True)
        q5 = DSModifier_quant(params={"bits": 5})
        q5.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + "#quant5_modifier/images/000000005802.jpg"
        ), "DSModifier_quant Failed"
        shutil.rmtree(ds_path + "#quant5_modifier", ignore_errors=True)

        shutil.rmtree(ds_path + "#jpg85_modifier#quant5_modifier", ignore_errors=True)
        jpg85_mas_q5 = DSModifier_quant(params={"bits": 5}, ds_modifier=jpg85)
        jpg85_mas_q5.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + "#jpg85_modifier#quant5_modifier/images/000000005802.jpg"
        ), "JPG+Quant modifier Failed"
        shutil.rmtree(ds_path + "#jpg85_modifier#quant5_modifier", ignore_errors=True)

        # DSModifier_rer
        params = {"initial_rer": 0.54, "rer": 0.1}
        blurimgmodif = DSModifier_rer(params=params)
        name = blurimgmodif.name
        shutil.rmtree(ds_path + f"#{name}", ignore_errors=True)
        blurimgmodif.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + f"#{name}/images/000000005802.jpg"
        ), "DSModifierBlur Failed"
        shutil.rmtree(ds_path + f"#{name}", ignore_errors=True)

        # DSModifier_rer
        # check default params
        blurimgmodif = DSModifier_rer()
        params = blurimgmodif.params
        name = f"rer{params['rer']}_modifier"
        shutil.rmtree(ds_path + f"#{name}", ignore_errors=True)
        blurimgmodif.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + f"#{name}/images/000000005802.jpg"
        ), "DSModifierBlur Failed"
        shutil.rmtree(ds_path + f"#{name}", ignore_errors=True)

        # DSModifier_snr
        noisemodif = DSModifier_snr()
        shutil.rmtree(ds_path + "#" + noisemodif.name, ignore_errors=True)
        noisemodif.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + "#" + noisemodif.name + "/images/000000005802.jpg"
        ), "DSModifierNoise Failed"
        shutil.rmtree(ds_path + "#" + noisemodif.name, ignore_errors=True)

        # DSModifier_blur
        shutil.rmtree(ds_path + "#blur2.0_modifier", ignore_errors=True)
        blur2 = DSModifier_blur(params={"sigma": 2.0})
        blur2.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + "#blur2.0_modifier/images/000000005802.jpg"
        ), "DSModifier_blur Failed"
        shutil.rmtree(ds_path + "#blur2.0_modifier", ignore_errors=True)

        # DSModifier_sharpness
        shutil.rmtree(ds_path + "#sharpness2.0_modifier", ignore_errors=True)
        sharpness2 = DSModifier_sharpness(params={"sharpness": 2.0})
        sharpness2.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + "#sharpness2.0_modifier/images/000000005802.jpg"
        ), "DSModifier_sharpness Failed"
        shutil.rmtree(ds_path + "#sharpness2.0_modifier", ignore_errors=True)

        # DSModifier_gsd
        name = f"gsd{0.3*2.0}_modifier"
        gsd = 0.60  # 0.3 * 2.0
        shutil.rmtree(ds_path + f"#gsd{gsd}_modifier", ignore_errors=True)
        gsd2 = DSModifier_gsd(params={"scale": 2.0, "interpolation": 2, "resol": 0.3})
        gsd2.modify(data_input=img_path)
        assert os.path.exists(
            ds_path + f"#gsd{gsd}_modifier/images/000000005802.jpg"
        ), "DSModifier_gsd Failed"
        shutil.rmtree(ds_path + f"#gsd{gsd}_modifier", ignore_errors=True)
