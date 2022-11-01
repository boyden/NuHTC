"""run.

Usage:
  print_metric.py --path=<n> [--exp=<n>]
  print_metric.py (-h | --help)
  print_metric.py --version

Options:
  -h --help             Show this string.
  --version             Show version.
  --path=<n>    Path.
  --exp=<n>    Experiment name.
"""


import docopt
import numpy as np
import os
import pandas as pd

def main(args):
    """
    This function returns the statistics reported on the PanNuke dataset, reported in the paper below:

    Gamper, Jevgenij, Navid Alemi Koohbanani, Simon Graham, Mostafa Jahanifar, Syed Ali Khurram,
    Ayesha Azam, Katherine Hewitt, and Nasir Rajpoot.
    "PanNuke Dataset Extension, Insights and Baselines." arXiv preprint arXiv:2003.10778 (2020).

    Args:
    Root path to the ground-truth
    Root path to the predictions
    Path where results will be saved

    Output:
    Terminal output of bPQ and mPQ results for each class and across tissues
    Saved CSV files for bPQ and mPQ results for each class and across tissues
    """

    eval_path = args['--path']
    conic_path = f'{eval_path}/conic_stats.csv'
    pannuke_path = f'{eval_path}/tissue_stats.csv'

    conic_metric = pd.read_csv(conic_path, index_col=0)
    pannuke_metric = pd.read_csv(pannuke_path, index_col=0)
    conic_dict = conic_metric.iloc[0].to_dict()
    pannuek_dict = pannuke_metric.iloc[-1].to_dict()
    aji, dice = conic_dict['aji'], conic_dict['dice']
    mPQ, PQ = pannuek_dict['PQ'], pannuek_dict['PQ bin']
    all_metrics = {
        'aji': [aji],
        'dice': [dice],
        'PQ': [PQ],
        'mPQ': [mPQ]
    }
    df = pd.DataFrame(all_metrics)
    df.to_csv(eval_path + '/format_stats.csv')
    df = df.to_string(index=False)
    print(df)



#####
if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='PanNuke Evaluation v1.0')
    main(args)
    basedir = '/home/bao/code/SoftTeacher'
    if not args['--exp']:
        exp_name = os.path.basename(args['--path'])[:-1]
    nuclei_stat_li = []
    mpq_li = []
    bpq_li = []
    for fold in range(1, 4):
        nuclei_stat = pd.read_csv(f'{basedir}/results/{exp_name}{fold}/tissue_stats.csv')
        nuclei_stat = nuclei_stat.drop(['Unnamed: 0'], axis=1)
        bpq_li.append(nuclei_stat.loc[19, ['PQ bin']].values[0])
        mpq_li.append(nuclei_stat.loc[19, ['PQ']].values[0])
        nuclei_stat_li.append(nuclei_stat)

    nuceli_total = nuclei_stat_li[0].loc[:, ['PQ', 'PQ bin']] + nuclei_stat_li[1].loc[:, ['PQ', 'PQ bin']] + nuclei_stat_li[2].loc[:, ['PQ', 'PQ bin']]
    nuceli_total = nuceli_total/3
    nuceli_total['Tissue name'] = nuclei_stat_li[0]['Tissue name']
    nuceli_total = nuceli_total[['Tissue name', 'PQ', 'PQ bin']]
    nuceli_total.loc[20] = ['std', np.std(mpq_li), np.std(bpq_li)]
    print(nuceli_total.round(4))
    nuclei_stat_li = []
    mpq_li = []
    for fold in range(1, 4):
        nuclei_stat = pd.read_csv(f'{basedir}/results/{exp_name}{fold}/class_stats.csv')
        nuclei_stat = nuclei_stat.drop(['Unnamed: 0'], axis=1)
        nuclei_stat_li.append(nuclei_stat)

    nuceli_total = nuclei_stat_li[0].loc[:, ['PQ']] + nuclei_stat_li[1].loc[:, ['PQ']] + nuclei_stat_li[2].loc[:, ['PQ', ]]
    nuceli_total = nuceli_total/3
    nuceli_total['Class Name'] = nuclei_stat_li[0]['Class Name']
    nuceli_total = nuceli_total[['Class Name', 'PQ']]
    print(nuceli_total.round(3))
