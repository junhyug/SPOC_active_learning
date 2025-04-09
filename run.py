import argparse
import os
import tempfile
import time

from engine import Engine
from utils import config_util
from utils.general_util import log, get_temp_dir, mkdir_p



def main(args):

    configs = config_util.load_configs(args.config_path)

    root_dir = get_temp_dir()
    if args.material:
        configs["data"]["material"] = args.material
    if args.print:
        configs["data"]["print_type"] = args.print

    data_config_values = "_".join([str(value).upper() for value in configs["data"].values() if value is not None])
    tempfile.tempdir = mkdir_p(os.path.join(root_dir, data_config_values))

    train_prefix = "%s-%s-" % (
        configs["model"]["name"].upper(),
        time.strftime("%Y%m%d-%H%M%S")
    )
    save_dir = tempfile.mkdtemp(
        suffix="-" + args.tag if args.tag else None,
        prefix=train_prefix
    )
    log.infov("Working Directory: {}".format(save_dir))
    engine = Engine(configs=configs, save_dir=save_dir, args_dict=vars(args))
    engine.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="",
                        help="path to a config")
    parser.add_argument("--save_dir", default="",
                        help="directory to save checkpointables")
    parser.add_argument("--tag", default="",
                        help="tag to discern training results")
    parser.add_argument("--acq", default=None,
                        help="acquisition function")
    parser.add_argument("--top_ratio", type=float, default=None,
                        help="ratio of the top items to consider")
    parser.add_argument("--material", default=None, help="material")
    parser.add_argument("--print", default=None, help="print type")
    args = parser.parse_args()

    main(args)
