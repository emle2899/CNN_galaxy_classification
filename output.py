import pandas as pd
import numpy as np
import os
from eda import read_images, targets, train_val_holdout, process_images, remove_dir

if __name__ == '__main__':
    # create train, validation, and holdout folders
    df = pd.read_csv('data/training_solutions_rev1.csv')
    paths = 'data/image_data/'

    # Allowing 70% agreement to classify galaxies
    # df = targets(df, p = 0.70)
    # df_new = df.filter(['GalaxyID','labels'], axis=1)
    # images, id_num = read_images(paths)
    #
    # df_id = pd.DataFrame({'GalaxyID': id_num})
    # data = pd.merge(df_id, df_new, on = 'GalaxyID', how = 'inner')
    # data.to_csv('data/galaxy_labels.csv', index=False)

    df_lbl = pd.read_csv('data/galaxy_labels.csv')
    train_val_holdout('data/image_data', 'data', df_lbl)

    # remove 'other' folder completely for training only on images with a consensus on label
    # paths_other = ('data/holdout/other','data/train/other', 'data/validation/other')
    # remove_dir(paths_other)
