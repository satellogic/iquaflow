import glob
import os
import shutil

from iquaflow.datasets import DSModifier, DSWrapper

current_path = os.path.dirname(os.path.realpath(__file__))
ds_name = "ds_coco_dataset"
base_ds = os.path.join(current_path, "test_datasets")
data_path = os.path.join(base_ds, ds_name)
data_path_geojson = os.path.join(base_ds, "ds_geo_dataset")
input_path = os.path.join(data_path, "images")
input_path_geojson = os.path.join(data_path_geojson, "images")
json_annotations_path = os.path.join(data_path, "coco_annotations.json")
geojson_annotations_path = os.path.join(data_path_geojson, "annots.geojson")

modifier_name = "base_modifier"


def check_modification(input_path: str, output_images_path: str) -> bool:
    original_images = [
        os.path.basename(f) for f in glob.glob(os.path.join(input_path, "*.jpg"))
    ]
    original_images.sort()
    modified_images = [
        os.path.basename(f)
        for f in glob.glob(os.path.join(output_images_path, "*.jpg"))
    ]
    modified_images.sort()
    mapped_list = [o == m for o, m in zip(original_images, modified_images)]
    return all(mapped_list)


class TestDSWrapper:
    def test_implicit_coco_initialization(self):
        ds_wrapper = DSWrapper(data_path=data_path)
        assert ds_wrapper.parent_folder == base_ds
        assert ds_wrapper.data_path == data_path, "root datset should be equal"
        assert (
            ds_wrapper.data_input == input_path
        ), "DSWrapper should find data input path"
        assert (
            ds_wrapper.json_annotations == json_annotations_path
        ), "DSWrapper should find json annotations"
        assert ds_wrapper.get_annotations() == [json_annotations_path]

    def test_explicit_coco_initialization(self):
        ds_wrapper = DSWrapper(
            data_input=input_path, json_annotations=json_annotations_path
        )
        assert ds_wrapper.parent_folder == base_ds
        assert ds_wrapper.data_path == data_path, "root datset should be equal"
        assert (
            ds_wrapper.data_input == input_path
        ), "DSWrapper should find data input path"
        assert (
            ds_wrapper.json_annotations == json_annotations_path
        ), "DSWrapper should find json annotations"

        # with pytest.raises(AssertionError):
        #    ds_wrapper = DSWrapper(data_input=input_path)

        # with pytest.raises(AssertionError):
        #    ds_wrapper = DSWrapper(json_annotations=json_annotations_path)
        assert ds_wrapper.get_annotations() == [json_annotations_path]

    def test_implicit_geojson_initialization(self):
        ds_wrapper = DSWrapper(data_path=data_path_geojson)
        assert ds_wrapper.parent_folder == base_ds
        assert ds_wrapper.data_path == data_path_geojson, "root datset should be equal"
        assert (
            ds_wrapper.data_input == input_path_geojson
        ), "DSWrapper should find data input path"
        assert (
            ds_wrapper.geojson_annotations == geojson_annotations_path
        ), "DSWrapper should find json annotations"
        assert ds_wrapper.get_annotations() == [geojson_annotations_path]

    def test_explicit_geojson_initialization(self):
        ds_wrapper = DSWrapper(
            data_input=input_path_geojson, geojson_annotations=geojson_annotations_path
        )
        assert ds_wrapper.parent_folder == base_ds
        assert ds_wrapper.data_path == data_path_geojson, "root datset should be equal"
        assert (
            ds_wrapper.data_input == input_path_geojson
        ), "DSWrapper should find data input path"
        assert (
            ds_wrapper.geojson_annotations == geojson_annotations_path
        ), "DSWrapper should find json annotations"
        assert ds_wrapper.get_annotations() == [geojson_annotations_path]

        # with pytest.raises(AssertionError):
        #    ds_wrapper = DSWrapper(data_input=input_path)

        # with pytest.raises(AssertionError):
        #    ds_wrapper = DSWrapper(json_annotations=json_annotations_path)

    def test_log_parameters(self):
        expected_log_params = {"ds_name": ds_name}
        expected_extended_log_params = {"ds_name": ds_name, "test": "test"}
        ds_wrapper = DSWrapper(data_path=data_path)

        assert (
            ds_wrapper.log_parameters() == expected_log_params
        ), "Base log parameters should be the same"

        ds_wrapper.set_log_parameters({"test": "test"})
        assert (
            ds_wrapper.log_parameters() == expected_extended_log_params
        ), "Extended log parameters should be the same"


class TestDSModifierSingleModification:
    def test_modifier_initialization(self):
        ds_modifier = DSModifier()
        assert ds_modifier.name == modifier_name
        assert ds_modifier._get_name() == modifier_name

    def test_dataset_modification(self):
        output_path = os.path.join(base_ds, ds_name + "#{}".format(modifier_name))
        output_images_path = os.path.join(output_path, "images")

        ds_modifier = DSModifier()
        modified_input_path, mod_path, parent_folder = ds_modifier.modify(
            data_path=data_path
        )

        assert mod_path == output_path
        assert os.path.exists(mod_path)
        assert modified_input_path == output_images_path
        assert os.path.exists(modified_input_path)
        assert check_modification(
            input_path, modified_input_path
        ), "modified images should be copied in {} and named the same".format(
            output_images_path
        )
        shutil.rmtree(mod_path)

    def test_log_parameters(self):
        ds_modifier = DSModifier()
        expected_log_params = {"modifier": "base_modifier"}
        assert ds_modifier.log_parameters() == expected_log_params


class TestDSModifierMultyModification:
    def test_modifier_multiple_initialization(self):
        ds_modifier = DSModifier(DSModifier())
        composed_names = "{}#{}".format(modifier_name, modifier_name)
        assert ds_modifier.name == modifier_name
        assert ds_modifier._get_name() == composed_names

    def test_dataset_modification(self):
        output_path = os.path.join(
            base_ds, ds_name + "#{}#{}".format(modifier_name, modifier_name)
        )
        output_images_path = os.path.join(output_path, "images")

        ds_modifier = DSModifier(DSModifier())
        modified_input_path, mod_path, parent_folder = ds_modifier.modify(
            data_path=data_path
        )

        assert mod_path == output_path
        assert os.path.exists(mod_path)
        assert modified_input_path == output_images_path
        assert os.path.exists(modified_input_path)
        assert check_modification(
            input_path, modified_input_path
        ), "modified images should be copied in {} and named the same".format(
            output_images_path
        )
        shutil.rmtree(mod_path)

    def test_log_parameters(self):
        ds_modifier = DSModifier(DSModifier())
        expected_log_params = {"modifier": "base_modifier#base_modifier"}
        assert ds_modifier.log_parameters() == expected_log_params


class TestDSWrapperAndDSModier:
    def test_initialization(self):
        output_path = os.path.join(base_ds, ds_name + "#{}".format(modifier_name))
        output_images_path = os.path.join(output_path, "images")

        ds_wrapper = DSWrapper(data_path=data_path)
        assert ds_wrapper.data_path == data_path, "root datset should be equal"
        assert (
            ds_wrapper.data_input == input_path
        ), "DSWrapper should find data input path"

        ds_modifer = DSModifier()
        mod_ds_wrapper = ds_wrapper.modify(ds_modifer)
        assert mod_ds_wrapper.data_path == output_path, "root datset should be equal"
        assert os.path.exists(mod_ds_wrapper.data_path)
        assert (
            mod_ds_wrapper.data_input == output_images_path
        ), "DSWrapper should find data input path"
        shutil.rmtree(mod_ds_wrapper.data_path)

    def test_log_parameters(self):
        ds_wrapper = DSWrapper(data_path=data_path)
        ds_modifer = DSModifier()
        mod_ds_wrapper = ds_wrapper.modify(ds_modifer)
        expected_extended_log_params = {
            "ds_name": ds_name + "#base_modifier",
            "modifier": "base_modifier",
        }
        assert (
            mod_ds_wrapper.log_parameters() == expected_extended_log_params
        ), "Extended log parameters should be the same"
        shutil.rmtree(mod_ds_wrapper.data_path)
