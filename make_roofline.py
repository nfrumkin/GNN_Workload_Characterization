import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse


parser = argparse.ArgumentParser(description="generate roofline model")
parser.add_argument('--csv', '-c', type=str, default="test.csv", help="input csv file path")
parser.add_argument('--png', '-p', type=str, default="test.png", help="output png file path")
parser.add_argument('--title', type=str, help="plot title")
parser.add_argument('--machine', '-m', type=str, choices=["TITAN-Xp", "1080ti", "v100", "a100"], default="titanxp", help="hardware type")
args = parser.parse_args()

"""Machine Specific Consts"""
if args.machine == "TITAN-Xp":
    # source: https://www.nvidia.com/en-us/titan/titan-xp/
    peak_flops = 12e+12 # TFLOPs
    peak_mem_band = 547.7e+9 # GB/s
elif args.machine == "1080ti":
    # source: https://www.anandtech.com/show/11180/the-nvidia-geforce-gtx-1080-ti-review
    peak_flops = 11.3e+12 # TFLOPs
    # source: https://www.nvidia.com/en-sg/geforce/products/10series/geforce-gtx-1080-ti/
    peak_mem_band = 484e+9 # GB/s
elif args.machine == "v100":
    # source: https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
    peak_flops = 14e+12 # TFLOPs single precision v100 for PCIe
    peak_mem_band = 900e+9 # GB/s for CoWoS Stacked HBM2
elif args.machine == "a100":
    # source: https://www.nvidia.com/en-us/data-center/a100/
    peak_flops = 19.5e+12 # TFLOPs single precision
    peak_mem_band = 1555e+9 # GB/s for PCIe
else:
    raise Exception("Invalid hardware type")

"""Run-Specific Consts"""
def parse_csv(csv_name):
    dram_read_transactions = []
    dram_write_transactions = []
    flops = []
    legend_names = []
    runtimes = []

    with open(csv_name, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dram_read_transactions.append(int(row['dramRead']))
            dram_write_transactions.append(int(row['dramWrite']))
            flops.append(int(row['flop_count']))
            legend_names.append(row['name'])
            runtimes.append(float(row['time']))
    return runtimes, legend_names, dram_read_transactions, dram_write_transactions, flops

def compute_ai(dram_read_transactions, dram_write_transactions, flop_count):
	return flop_count / ((dram_read_transactions + dram_write_transactions)*32)


if __name__ == "__main__":

    # FLOP : byte
    # 32 is size of each memory transaction

    runtimes, legend_names, dram_read, dram_write, flops = parse_csv(args.csv)

    max_ai = 2
    for i in range(0,len(legend_names)):
        ai = compute_ai(dram_read[i], dram_write[i], flops[i])
        print(ai)
        max_ai = max(max_ai, ai)
        flops_per_second = flops[i]/runtimes[i]
        plt.scatter(np.array([ai]),np.array([flops_per_second]), color='r', label=legend_names[i])

    granularity = 20 # number of points to compute
    # range to display
    ai_range = np.linspace(0,max_ai,granularity)

    # compute computational roof
    comp_roof = [peak_flops for i in range(0,granularity)]
    # compute memory bandwidth ceiling
    bnd_ceiling = [ ai_range[i]*peak_mem_band for i in range(0,granularity)]

    # plot roofline
    roofline = [ min(bnd_ceiling[i], comp_roof[i]) for i in range(0,granularity)]
    plt.loglog(ai_range, roofline)
    if args.title is None:
        title = "Roofline for " + args.machine + "     " + str(peak_flops/1e+12) + " Tflop/s     " + str(peak_mem_band/1e+9) + " GB/s"
    else:
        title = args.title
    plt.title(title)
    # plt.legend()
    plt.xlabel("Arithmetic Intensity")
    plt.ylabel("GFLOP/s")
    plt.legend()
    plt.savefig(args.png)

