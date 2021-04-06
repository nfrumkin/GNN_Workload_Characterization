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
parser.add_argument('--folder', '-f', type=str, help='names of log folder', required=True)
parser.add_argument('--out', '-o', type=str, default="stats.csv", help='names of output csv')
args = parser.parse_args()

def compute_ai(dram_read_transactions, dram_write_transactions, flop_count):
    return int(float(flop_count) / ((float(dram_read_transactions) + float(dram_write_transactions))*32))

def compute_ai_params(params, flop_count):
    return int(float(flop_count) / ((float(params))*32))

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
    csv_name = args.out
    
    fields = ["name", "time", "flop_count", "dramRead", "dramWrite", "flops", "ai"]
    csvfile = open(csv_name, 'w')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

    run_folder = args.folder
    run_dict = {}

    # field: name
    run_dict["name"] = "pygcn"

    # field: time
    run_dict["time"] = float(read_head(run_folder+"/time.txt"))

    filename = run_folder+"/nvprof.out"

    # field: flop count
    run_dict["flop_count"] =  read_nvprof_metric(filename, "flop_count_sp")


    # field: dramRead
    run_dict["dramRead"] = read_nvprof_metric(filename, "dram_read_transactions")
    
    #field: dramWrite
    run_dict["dramWrite"] = read_nvprof_metric(filename, "dram_write_transactions")

    dram_transactions = run_dict["dramRead"] + run_dict["dramWrite"]

    # field: flops
    run_dict["flops"] = run_dict["flop_count"]/run_dict["time"]

    # field: ai
    run_dict["ai"] = compute_ai_params(dram_transactions, run_dict["flop_count"])
    csvwriter.writerow(run_dict.values())

if __name__ == "__main__":
    main(sys.argv)
