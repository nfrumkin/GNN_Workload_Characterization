#!/usr/bin/env python3

import sys 
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
from os import listdir
from os.path import isfile, join
import re

parser = argparse.ArgumentParser(description="parse nvprof log files")
parser.add_argument('--folder', type=str, help='name of log folder', required=True)
parser.add_argument('--file', type=str, nargs='+', help='names of log files')
parser.add_argument('--csv', '-c', type=str, default="stats.csv", help='names of output csv')
args = parser.parse_args()

def compute_ai(dram_transactions, flop_count):
    return float(float(flop_count) / (float(dram_transactions)*32))

def compute_ai_params(params, flop_count):
    return float(flop_count) / (float(params)*32)

def read_nvprof_metric(inputFile, metric):
    # sum over all instances of given metric to get total value
    value = 0

    with open(inputFile, "r") as f: 
        data = f.readlines()

    for line in data:
        if metric in line:
            line = line.split(",")
            value = value + int(line[-1])

    return value

def read_head(filename):
     with open(filename, "r") as f:
            return f.readline()[:-1] # exclude ending "\n"

def main(argv):
    csv_name = args.csv

    if args.file is None: # infer run names from run id files
        id_file = args.folder+"/run_ids.txt"
        with open(id_file, "r") as f:
            run_names = f.readlines()
            run_names = [run[:-1] for run in run_names]
    else:
        run_names = list(args.file)
    
    run_names.sort()
    
    fields = ["name", "time", "flop_count", "dram_accesses", "flops", "ai"]
    csvfile = open(csv_name, 'w')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

    for run in run_names:
        if args.file is not None:
            run_folder = args.folder
        else:
            run_folder = args.folder+"/"+run
        run_dict = {}

        # field: name
        run_dict["name"] = run

        # field: time
        run_dict["time"] = read_head(run_folder+"/time.txt")

        # field: flop count
        run_dict["flop_count"] = read_nvprof_metric(run, "flop_count_sp") + read_nvprof_metric(run, "flop_count_dp")

        # field: params
        run_dict["dram_accesses"] = read_nvprof_metric(run, "dram_read_transactions") + read_nvprof_metric(run, "dram_write_transactions")
        
        # field: flops
        run_dict["flops"] = float(run_dict["flop_count"])/float(run_dict["time"])

        # field: ai
        run_dict["ai"] = compute_ai(run_dict["dram_accesses"], run_dict["flop_count"])

        csvwriter.writerow(run_dict.values())

if __name__ == "__main__":
    main(sys.argv)
