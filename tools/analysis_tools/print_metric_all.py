"""run.

Usage:
  print_metric.py --path=<n>
  print_metric.py (-h | --help)
  print_metric.py --version

Options:
  -h --help             Show this string.
  --version             Show version.
  --ath=<n>    Path.
"""


import docopt
import os
import pandas as pd

def metric_log(eval_path):
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
    return df



#####
if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='PanNuke Evaluation v1.0')
    main(args)
