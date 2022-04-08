import argparse
import itertools
import os
from typing import Any, Dict, List, Tuple

# import json


def read_args() -> Any:
    parser = argparse.ArgumentParser(description="Regressor .cfg generator")
    parser.add_argument(
        "--cfg_folder", type=str, default="cfgs/"
    )  # where to save these cfg files
    # [RUN]
    parser.add_argument("--trainid", type=str, default="test")
    parser.add_argument("--resume", action="store_true")
    # [PATHS]
    parser.add_argument(
        "--trainds",
        type=str,
        default="/home/Imatge/projects/satellogic/iquaflow-/tests/test_datasets/AerialImageDataset",
    )
    parser.add_argument(
        "--traindsinput",
        type=str,
        default="/home/Imatge/projects/satellogic/iquaflow-/tests/test_datasets/AerialImageDataset/train/images",
    )
    parser.add_argument(
        "--valds",
        type=str,
        default="/home/Imatge/projects/satellogic/iquaflow-/tests/test_datasets/AerialImageDataset",
    )
    parser.add_argument(
        "--valdsinput",
        type=str,
        default="/home/Imatge/projects/satellogic/iquaflow-/tests/test_datasets/AerialImageDataset/test/images",
    )
    parser.add_argument("--outputpath", type=str, default="tmp-")
    # [HYPERPARAMS]
    parser.add_argument("--num_regs", type=int, nargs="+", action="append")
    parser.add_argument(
        "--modifier_params",
        type=str,
        nargs="+",
        default=['{"sigma": np.linspace(1.0, 2.5, 50)}'],
    )
    parser.add_argument("--epochs", type=int, nargs="+", default=[200])
    parser.add_argument("--num_crops", type=int, nargs="+", default=[20])
    parser.add_argument("--splits", type=float, nargs="+", action="append")
    parser.add_argument("--input_size", type=int, nargs="+", action="append")
    parser.add_argument("--batch_size", type=int, nargs="+", default=[32])
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--momentum", type=float, nargs="+", default=[0.9])
    parser.add_argument("--weight_decay", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--soft_threshold", type=float, nargs="+", default=[0.3])
    parser.add_argument("--workers", type=int, default=8)  # unique
    parser.add_argument("--data_shuffle", type=bool, default=True)  # unique
    args = parser.parse_args()
    # optional (default) params for list of lists (params that are lists per se)
    if args.num_regs is None:
        args.num_regs = [[50]]
    if args.splits is None:
        args.splits = [[0.8, 0.2]]
    if args.input_size is None:
        args.input_size = [[128, 128]]
    return args


def dict2cfg(dict_set: Dict[str, Any], cfg_path: str, superargs: List[Any]) -> bool:
    print(cfg_path)
    cfg_file = open(cfg_path, "w")
    for superarg in superargs:
        cfg_file.writelines(superarg + "\n")
        for key in list(dict_set[superarg].keys()):
            line_str = key + "=" + str(dict_set[superarg][key])
            cfg_file.writelines(line_str + "\n")
    cfg_file.close()
    return os.path.exists(cfg_path)


def args2powersetdict(
    args: Any,
    powerset_args: List[Any],
    args_unique: List[Any],
    dict_args_cfg_empty: Dict[str, Any],
) -> Tuple[Any, Any]:

    # get combinations
    dicts_sets = []
    names_sets = []
    powerset = [getattr(args, arg) for arg in powerset_args]
    combinations = list(itertools.product(*powerset))

    # loop: create dict of args with values
    for pset in combinations:
        cfg_dict = dict_args_cfg_empty  # current cfg
        trainid_combi = ""
        # COMBINATION ARGUMENTS
        for idx, arg in enumerate(powerset_args):
            for superarg in list(cfg_dict.keys()):
                if arg in list(cfg_dict[superarg].keys()):
                    cfg_dict[superarg][arg] = pset[idx]
                    if arg in args_jloads:
                        cfg_dict[superarg][arg] = pset[idx].replace(
                            "'", ""
                        )  # remove string "\'" usage
            # set trainid name suffix
            if arg in args_jloads:  # modifier_params case
                # dict_param = json.loads(pset[idx].split('"')[1].replace("np.linspace", "").replace("(", '"').replace(")", '"'))
                # key = list(dict_param.keys())[0]
                key = pset[idx].split('"')[1]
                trainid_combi += "_"
                trainid_combi += str(key)
            else:
                trainid_combi += "_"
                trainid_combi += arg.replace("_", "")
                trainid_combi += str(pset[idx])
        # UNIQUE ARGUMENTS
        for idx, arg in enumerate(args_unique):
            for superarg in list(cfg_dict.keys()):
                if arg in list(cfg_dict[superarg].keys()):
                    cfg_dict[superarg][arg] = getattr(args, arg)
        # replace tokens to set name
        trainid_combi = (
            trainid_combi.replace("[", "")
            .replace("]", "")
            .replace(" ", "")
            .replace(",", "-")
        )
        # append lists of dict (configparse combinations) and trainids as names of files
        dicts_sets.append(cfg_dict)
        names_sets.append(trainid_combi)
    return dicts_sets, names_sets


if __name__ == "__main__":
    # parse args and prepare output folder for cfgs
    args = read_args()
    os.makedirs(args.cfg_folder, exist_ok=True)
    # dummy cfg
    dict_args_cfg_empty = {
        "[RUN]": {"trainid": None, "resume": None},
        "[PATHS]": {
            "trainds": None,
            "traindsinput": None,
            "valds": None,
            "valdsinput": None,
            "outputpath": None,
        },
        "[HYPERPARAMS]": {
            "num_regs": None,
            "modifier_params": None,
            "epochs": None,
            "num_crops": None,
            "splits": None,
            "input_size": None,
            "batch_size": None,
            "lr": None,
            "momentum": None,
            "weight_decay": None,
            "soft_threshold": None,
            "workers": None,
            "data_shuffle": None,
        },
    }
    # get trainid prefix
    prefix = args.trainid
    dataset_name = os.path.basename(args.trainds)
    trainid_prefix = prefix + "_" + dataset_name

    # check args types
    args_ignore = ["cfg_folder"]
    args_jloads = ["modifier_params"]
    args_unique = (
        list(dict_args_cfg_empty["[RUN]"].keys())
        + list(dict_args_cfg_empty["[PATHS]"].keys())
        + ["workers", "data_shuffle"]
    )
    args_lists = ["num_regs", "splits", "input_size"]  # args that per se are lists
    args_single = [
        arg
        for arg in vars(args)
        if arg not in args_unique + args_ignore and len(getattr(args, arg)) == 1
    ]
    args_multi = [
        arg
        for arg in vars(args)
        if arg not in args_unique + args_ignore and len(getattr(args, arg)) > 1
    ]

    # get list of dicts and names for every combination permutation of arguments
    powerset_args = (
        args_multi + args_single
    )  # args that may vary according to argparsed
    dicts_sets, names_sets = args2powersetdict(
        args, powerset_args, args_unique, dict_args_cfg_empty
    )

    # add original prefix with trainid_combi
    for idx, dict_set in enumerate(dicts_sets):
        names_sets[idx] = trainid_prefix + names_sets[idx]
        dicts_sets[idx]["[RUN]"]["trainid"] = names_sets[idx]

    # write cfg file for each dict
    for idx, dict_set in enumerate(dicts_sets):
        cfg_path = os.path.join(args.cfg_folder, names_sets[idx] + ".cfg")
        superargs = list(dict_set.keys())
        success = dict2cfg(dict_set, cfg_path, superargs)
