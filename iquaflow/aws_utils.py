import os
from typing import Union

import boto3
from joblib import Parallel, delayed


def treat_path_string_start(ds_path: str) -> str:
    """
    remove initial ./ from ds_path

    Args:
        ds_path: Input dataset path

    Returns:
        A transformed ds_path
    """
    for i in range(5):
        ds_path = (
            ds_path[1:]
            if any(ds_path.startswith(s) for s in [".", os.sep])
            else ds_path
        )
    return ds_path


def upload_objects(
    bucket_name: str = "iq-airport-use-case",
    root_path: str = "/scratch/SATE00_MFSR00/DATA_SOURCES/",
    root_pth_bucket: str = "DATA_SOURCES/",
    upload_num_threads: int = 10,
) -> None:
    """
    Upload the contents of a folder directory

    Args:
        bucket_name: the name of the s3 bucket
        root_pth_bucket: the folder path in the s3 bucket
        root_path: a relative or absolute directory path in the local file system
    """

    s3_resource = boto3.resource("s3", region_name="eu-west-1")

    try:

        my_bucket = s3_resource.Bucket(bucket_name)

        for pth, subdirs, files in os.walk(root_path):

            directory_name = pth.replace(root_path, "")

            if directory_name.startswith(os.sep):
                directory_name = directory_name[1:]

            if len(files) > 0:

                src = os.path.join(pth, files[0])
                dst = os.path.join(root_pth_bucket, directory_name, files[0])
                print("Uploading data to S3... i.e.")
                print(f"SRC > {src}")
                print(f"DST > {dst}")

                Parallel(n_jobs=upload_num_threads, prefer="threads", verbose=10)(
                    delayed(my_bucket.upload_file)(
                        os.path.join(pth, base_fn),
                        os.path.join(root_pth_bucket, directory_name, base_fn),
                    )
                    for base_fn in files
                )

    except Exception as err:

        print("Error:", err)


def download_s3_folder(
    bucket_name: str, s3_folder: str, local_dir: Union[str, None] = None
) -> None:
    """
    Download the contents of a folder directory
    function downloaded from stackoverflow

    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = (
            obj.key
            if local_dir is None
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        )
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue
        bucket.download_file(obj.key, target)


def download_s3_filename(
    bucket_name: str, s3_filename: str, local_filename: Union[str, None] = None
) -> str:
    """
    Download file from bucket

    Args:
        bucket_name: the name of the s3 bucket
        s3_filename: the filename in the s3 bucket
        local_filename: The local filename in the local file system
    
    Returns:
        The local filename in the local file system
    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(s3_filename, local_filename)
    return str(local_filename)
