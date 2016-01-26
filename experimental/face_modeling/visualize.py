import datetime

import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

def plot_diagnostics(df, active_features, output_dir):
    df['date'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x).date())
    total_days = len(df.groupby('date'))
    i=1
    plt.figure(figsize=(20,10))
    X = fit_pca(df, active_features)
    for date, df_g in df.groupby('date'):
        plt.subplot(total_days, 1, i)
        plt.scatter(list(xrange(len(df_g))), df_g['noseX_raw'], c=df_g['class'])
        plt.title(date)
        i += 1
    print 'Saving plots to :', output_dir
    plt.savefig('%s-plots.png' % (output_dir))
    plt.figure()
    plt.scatter(X[:,0], X[:,1],c=df['class'])
    plt.savefig('%s-pca-plot.png' % output_dir)
 
def fit_pca(df, active_features, k = 2):
    pca = PCA(n_components=k)
    X = pca.fit_transform(df[active_features])
    return X



