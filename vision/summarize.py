import os
import json
import glob

for dataset in ['resisc45','dtd','cars','cifar100','flowers102','food101','sun397']:
#for dataset in ['sun397']:    
    model='vit-base'
    methods = ['base','ours','ourssvd1','ourssvdr','basesvd1']
    #methods = ['ourssvd1','ourssvdr','ourssvd1v2','ourssvdrv2']

    rs = [4,16,32]

    dirs = []
    results = {}
    for method in methods:
        folders = glob.glob(f'./experiment/{dataset}/{model}*_{method}')
        for folder in folders:
            dirs.append(folder.split('/')[-1])
        
        results[method] = {}


    for method in methods:
        for r in rs:
            results[method]['r'+str(r)] = {}
            for i in range(8):
                results[method]['r'+str(r)][f'scale{2+4*i}'] = []

    for dir in sorted(dirs):
        try:
            split = dir.split('_')
            scale, r, method = split[-3:]
            scale = scale.replace('alpha','scale')
        except:
            continue

        for seed in range(1,6):
            json_file = os.path.join(f'./experiment/{dataset}',dir,f'final_eval_results_{seed}.json')
            with open(json_file, 'r') as f:
                data = json.load(f)
                acc = float(data['eval_accuracy'])

            results[method][r][scale].append(acc)
            if len(results[method][r][scale]) == 5:
                results[method][r][scale] = sum(results[method][r][scale]) / 5

    import matplotlib.pyplot as plt

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c2','#7f7f7f','#bcbd22']
    colors = {'base': colors[0], 'ours': colors[1], 'ourssvd1': colors[2], 'basesvd1': colors[3], 'ourssvdr': colors[4], 'ourssvd1v2': colors[5], 'ourssvdrv2': colors[6]}


    for rank in rs:
        for method in results.keys():
            for r in results[method].keys():
                for scale in results[method][r].keys():
                    if r != f'r{rank}': continue
                    acc = results[method][r][scale]

                    plt.scatter(float(scale.replace('scale','')),acc,color=colors[method], marker='x', label=f'{method}_{r}')

        plt.grid(True, alpha=0.5)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys())
        plt.savefig(f'experiment/{dataset}_{model}_r{rank}.png')

        plt.cla()
        plt.clf()