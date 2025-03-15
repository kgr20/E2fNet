import os
from pathlib import Path
from rich import print

import argparse
import utils
import data_cfg

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, required=True, help="Dataset name (NODDI, Oddball, CNEPFL)")

if __name__=="__main__":
    args = parser.parse_args()
    print(args)

    assert args.data_name in data_cfg.raw_data_roots.keys(), \
        f"{args.data_name=} not in {list(data_cfg.raw_data_roots.keys())}"
    
    data_dir = Path(data_cfg.raw_data_roots[args.data_name])
    print(f"Raw data from: {data_dir}")
    
    if args.data_name == "CNEPFL":
        individuals = [x for x in os.listdir(data_dir) 
                       if "sub-" in x and os.path.isdir(data_dir/x)]
    elif args.data_name == "NODDI":
        individuals = os.listdir(data_dir/"EEG")
    else:
        individuals = [x for x in os.listdir(data_dir) 
                       if os.path.isdir(data_dir/x)]

    individuals = sorted(individuals)
    print(f"Individual names: {individuals}")

    dest_dir = Path(data_cfg.processed_data_roots[args.data_name])
    os.makedirs(dest_dir)
    print(f"Save h5 data to: {dest_dir}")

    if args.data_name == "CNEPFL":
        runs = [1]
        utils.save_h5_data_CNEPFL(
            individuals=individuals, runs=runs, processing_cfg=data_cfg.processing_cfg, 
            data_dir=data_dir, dest_dir=dest_dir)
        
    elif args.data_name == "NODDI":
        individuals = os.listdir(data_dir/"EEG")
        utils.save_h5_data_NODDI(
            individuals=individuals, processing_cfg=data_cfg.processing_cfg, 
            data_dir=data_dir, dest_dir=dest_dir)
        
    else:
        tasks = [1, 2]
        runs = [1, 2, 3]
        utils.save_h5_data_Oddball(
            individuals=individuals, processing_cfg=data_cfg.processing_cfg, tasks=tasks, 
            runs=runs, data_dir=data_dir, dest_dir=dest_dir)