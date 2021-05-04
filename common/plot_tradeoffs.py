import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import math
import os
import pandas as pd
parser = argparse.ArgumentParser(description="generate accuracy/latency tradeoff curve")
parser.add_argument('--csv', '-c', type=str, default="test.csv", nargs='+', help="input csv file path")
parser.add_argument('--machine', '-m', type=str, choices=["TITAN-Xp", "1080ti", "v100", "a100"], default="titanxp", help="hardware type")
parser.add_argument('--plotdir', '-o', type=str, default=".", help="plot output dir")
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

def create_dict_from_csv(filename, key, value):
    temp_dict = dict()
    csvfile = open(filename, mode='r')

    reader = csv.DictReader(csvfile)

    for row in reader:
        temp_dict[row[key]] = row[value]
    
    return temp_dict

def dist_to_knee(FLOPS, AI):
    return math.sqrt((peak_flops-FLOPS)**2 + (peak_flops/peak_mem_band - AI)**2)

"""Run-Specific Consts"""
def parse_csv(csv_name):
    latencies = []
    knee_dist_vals = []
    legend_names = []
    flops = []
    param_count = []
    ais = []

    with open(csv_name, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            flop_count = float(row['flops'])
            time = float(row['time'])
            name = row['name']
            ai = float(row['ai'])

            latencies.append(time)
            flops.append(flop_count/time)

            if '@' in name:
                name = name.split('@')[1].split("_")[0]
            else:
                name = name.split("_")[0]
            legend_names.append(name)
            knee_dist_vals.append(dist_to_knee(flop_count/time, ai))
            param_count.append(float(row["params"]))
            ais.append(ai)
    return legend_names, latencies, knee_dist_vals, flops, param_count,ais

def compute_ai(dram_read_transactions, dram_write_transactions, flop_count):
	return flop_count / ((dram_read_transactions + dram_write_transactions)*32)


if __name__ == "__main__":

    plotdir = args.plotdir
    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    # FLOP : byte
    # 32 is size of each memory transaction
    latencies_1080ti = ["15ms", "22ms", "27ms"]
    latencies_v100 = ["6ms","9ms","11ms"]

    master_dict = dict()
    metrics = ["flops", "param_count", "latency"]

    top1_error_dict = create_dict_from_csv("model_error.csv", "name", "top1_error")

    for csv_name in args.csv:
        legend_names, latencies, knee_dist, flops, param_count,ais = parse_csv(csv_name)
    
        for i in range(0,len(legend_names)):
            nm = legend_names[i]
            if nm in master_dict.keys():
                # add run to val_dict
                master_dict[nm]["flops"].append(flops[i])
                master_dict[nm]["latency"].append(latencies[i])
                master_dict[nm]["knee_dist"].append(knee_dist[i])
                master_dict[nm]["param_count"].append(param_count[i])
                master_dict[nm]["ai"].append(ais[i])
            else:
                # generate new dict for new name
                val_dict = dict()
                val_dict["flops"] = [flops[i]]
                val_dict["latency"] = [latencies[i]] 
                val_dict["knee_dist"] = [knee_dist[i]]
                val_dict["param_count"] = [param_count[i]]
                val_dict["ai"] = [ais[i]]
                # report top1 accuracy from top1 error
                val_dict["top1"] = 100 - float(top1_error_dict[nm])
                master_dict[nm] = val_dict

    for x_value in metrics:
        for name in master_dict.keys():
            if name[0:8] == "densenet":
                plt.scatter(np.mean(master_dict[name][x_value]),np.mean(master_dict[name]["top1"]), label=name)
            elif name[0:3] == "vgg":
                plt.scatter(np.mean(master_dict[name][x_value]),np.mean(master_dict[name]["top1"]), label=name, marker='^')
            elif name[0:9] == "mobilenet":
                plt.scatter(np.mean(master_dict[name][x_value]),np.mean(master_dict[name]["top1"]), label=name, marker='>')
            elif name[0:6] == "resnet":
                plt.scatter(np.mean(master_dict[name][x_value]),np.mean(master_dict[name]["top1"]), label=name, marker='d')
            elif name in latencies_1080ti:
                plt.scatter(np.mean(master_dict[name][x_value]),np.mean(master_dict[name]["top1"]), label=name+"_1080ti", marker='v')
            elif name in latencies_v100:
                plt.scatter(np.mean(master_dict[name][x_value]),np.mean(master_dict[name]["top1"]), label=name+"_v100", marker='<')
            else:
                plt.scatter(np.mean(master_dict[name][x_value]),np.mean(master_dict[name]["top1"]), label=name, marker='*')

        title = x_value+"_accuracy_tradeoff"
        plt.title(title)

        plt.xlabel(x_value)
        plt.ylabel("Top1 Accuracy")
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig(args.plotdir+"/"+title+".png", bbox_inches="tight")


        plt.close()

    for name in master_dict.keys():
        if name[0:8] == "densenet":
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"]), label=name)
        elif name[0:3] == "vgg":
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"]), label=name, marker='^')
        elif name[0:9] == "mobilenet":
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"]), label=name, marker='>')
        elif name[0:6] == "resnet":
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"]), label=name, marker='d')
        elif name in latencies_1080ti:
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"]), label=name+"_1080ti", marker='v')
        elif name in latencies_v100:
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"]), label=name+"_v100", marker='<')
        else:
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"]), label=name, marker='*')

    title = "Roofline for " + args.machine + "     " + str(peak_flops/1e+12) + " Tflop/s     " + str(peak_mem_band/1e+9) + " GB/s"
    plt.title(title)
    plt.xlabel("Arithmetic Intensity")
    plt.ylabel("FLOPs")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    title = "roofline_no_roof"
    plt.savefig(args.plotdir+"/"+title+".png", bbox_inches="tight")

    max_ai = np.max(master_dict[name]["ai"])
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

    title = "roofline"
    plt.savefig(args.plotdir+"/"+title+".png", bbox_inches="tight")
    plt.close()

    for name in master_dict.keys():
        if name[0:8] == "densenet":
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"])*(master_dict[name]["top1"]/100), label=name)
        elif name[0:3] == "vgg":
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"])*(master_dict[name]["top1"]/100), label=name, marker='^')
        elif name[0:9] == "mobilenet":
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"])*(master_dict[name]["top1"]/100), label=name, marker='>')
        elif name[0:6] == "resnet":
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"])*(master_dict[name]["top1"]/100), label=name, marker='d')
        elif name in latencies_1080ti:
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"])*(master_dict[name]["top1"]/100), label=name+"_1080ti", marker='v')
        elif name in latencies_v100:
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"])*(master_dict[name]["top1"]/100), label=name+"_v100", marker='<')
        else:
            plt.scatter(np.mean(master_dict[name]["ai"]),np.mean(master_dict[name]["flops"])*(master_dict[name]["top1"]/100), label=name, marker='*')

    title = "Roofline for " + args.machine + "     " + str(peak_flops/1e+12) + " Tflop/s     " + str(peak_mem_band/1e+9) + " GB/s"
    plt.title(title)

    plt.xlabel("Arithmetic Intensity")
    plt.ylabel("FLOPs*Accuracy")
    # handles,labels = plt.gca().get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # plt.legend(handles, labels)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    title = "roofline_acc_no_roof"
    plt.savefig(args.plotdir+"/"+title+".png", bbox_inches="tight")

    max_ai = np.max(master_dict[name]["ai"])
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
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    title = "roofline_acc"
    plt.savefig(args.plotdir+"/"+title+".png", bbox_inches="tight")
    plt.close()