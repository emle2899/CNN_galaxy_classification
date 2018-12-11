import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from eda import read_images, targets, train_val_holdout, process_images

def randomForest(x,y, labels):
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)

    rf.fit(X_train, y_train)
    predicted = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    print(f'score estimate: {rf.oob_score_:.3}')
    print(f'Mean accuracy score: {accuracy:.3}')

if __name__ == '__main__':
    paths = '/Users/emily/Documents/galvanize/code_sprints/capstone2/data/small/'
    df = pd.read_csv('/Users/emily/Documents/galvanize/code_sprints/capstone2/data/training_solutions_rev1.csv')
    images, id_num = read_images(paths)
    input_data = process_images(images)

    df = targets(df, p = 0.70)
    df_new = df.filter(['GalaxyID','labels'], axis=1)

    # df_alt for random forest
    df_alt = df_new[:2500]
    df_alt = df_alt.replace({'labels' : {'other':0, 'star':1, 'irregular':2, 'barred_spiral':3, 'edge_view_spiral':4, 'spiral':5, 'elliptical':6}})

    count = len(input_data)
    arr = np.zeros(shape=(count,10800))
    for i in range(count):
        data_flat = input_data[i].T.flatten()
        arr[i] = data_flat

    pca = PCA(n_components=2500) #pca object
    X_pca = pca.fit_transform(arr)

    y = df_alt.pop('labels')
    randomForest(X_pca, y, df['labels'])
