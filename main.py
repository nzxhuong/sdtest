import argparse
import copy
import logging
import os
import shutil
import sys
import traceback

import numpy as np
import torch
import yaml

from runners.image_editing import Diffusion


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--sample", action="store_true", help="Whether to produce samples from the model")
    parser.add_argument("-i", "--image_folder", type=str, default="images", help="The folder name of samples")
    parser.add_argument("--ni", action="store_true", help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--mask_image_file", type=str, required=True)
    parser.add_argument("--sample_step", type=int, default=3, help="Total sampling steps")
    parser.add_argument("--t", type=int, default=400, help="Sampling noise scale")
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"level {args.verbose} not supported")

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(args.exp, exist_ok=True)
    args.image_folder = os.path.join(args.exp, args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # logging.info(f"Using device: {device}")
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    print(">" * 80)
    logging.info(f"Exp instance id = {os.getpid()}")
    if args.comment != "":
        logging.info(f"Exp comment = {args.comment}")
    logging.info(f"Config: ")
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            logging.info(f"  {key}:")
            for k, v in vars(value).items():
                logging.info(f"    {k} = {v}")
        else:
            logging.info(f"  {key} = {value}")
    print("<" * 80)

    try:
        runner = Diffusion(args, config)
        runner.image_editing_sample()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
