# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import os
import pprint
import sys
import yaml

try:
    import submitit
except ModuleNotFoundError:
    submitit = None

from train import main as app_main
from utils.distributed import cleanup_distributed

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs')
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-lauch several config files')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='num. nodes to request for job')
parser.add_argument(
    '--tasks-per-node', type=int, default=1,
    help='num. procs to per node')
parser.add_argument(
    '--time', type=int, default=4300,
    help='time in minutes to run job')


class Trainer:

    def __init__(self, fname='configs.yaml', load_model=None):
        self.fname = fname
        self.load_model = load_model

    def __call__(self):
        if submitit is None:
            raise ModuleNotFoundError(
                "SLURM launching requires `pip install submitit`."
            )
        fname = self.fname
        load_model = self.load_model
        logger.info(f'called-params {fname}')

        # -- load script params
        params = None
        with open(fname, 'r') as y_file:
            params = yaml.safe_load(y_file)
            logger.info('loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)

        resume_preempt = False if load_model is None else load_model
        environment = submitit.JobEnvironment()
        os.environ.setdefault("MASTER_ADDR", environment.hostnames[0])
        os.environ.setdefault("MASTER_PORT", "40112")
        try:
            app_main(args=params, resume_preempt=resume_preempt)
        finally:
            cleanup_distributed()

    def checkpoint(self):
        if submitit is None:
            raise ModuleNotFoundError(
                "SLURM checkpointing requires `pip install submitit`."
            )
        fb_trainer = Trainer(self.fname, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch():
    if submitit is None:
        raise ModuleNotFoundError(
            "SLURM launching requires `pip install submitit`."
        )
    executor = submitit.AutoExecutor(
        folder=os.path.join(args.folder, 'job_%j'),
        slurm_max_num_timeout=20)
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_mem_per_gpu='55G',
        timeout_min=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=10,
        gpus_per_node=args.tasks_per_node)

    config_fnames = [args.fname]

    jobs, trainers = [], []
    with executor.batch():
        for cf in config_fnames:
            fb_trainer = Trainer(cf)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        print(job.job_id)


if __name__ == '__main__':
    args = parser.parse_args()
    launch()
