# GNN_Workload_Characterization

### PyGAT Usage
1. Clone PyGAT repo
2. Test install by training `python train.py`
3. copy .pkl file and test.py into the PyGAT dir
4. collect runtime by running test.py on trained-model.pkl using: `python test.py --pkl_file gat.pkl --time_file time.txt`
5. collect metrics by running profiler on trained-model.pkl using: `python test.py --pkl_file gat.pkl` (note: do not specify time file)

### PyGCN Usage
Same as PyGAT above, just with pygcn repo and pygcn folder

## Profiler
collect profiler metrics by running: 'python3 profiler.py --pkl_file gat.pkl'
