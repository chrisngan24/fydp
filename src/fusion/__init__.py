import sklearn.neighbors as neighbors
import pandas as pd

def compare_df(df_1, df_2):
    """
    Returns the first one as the more dense
    and the second as the more sparse dataset
    """
    if len(df_1) >= len(df_2):
        return (df_1, df_2)
    return (df_2, df_1)

def fuse_data(df_base, df_inject, distance_col):
    """
    Assumes one data set is more dense than the other, and will inject
    the sparse dataset into the dense one
    %df_base% - the data frame of the base
    %df_inject% - the data frame that we wish to inject data into
    %distnace_col% - the column that represents the distance metric 
    """
    index_b = df_base.index
    distance_b = df_base[distance_col]
    df_base['dummy_index'] = index_b
    # slower one
    index_i = df_inject.index
    distance_i = df_inject[distance_col].reshape(len(df_inject),1)
    df_inject['dummy_index'] = index_i
    fused_index = []
    ann = neighbors.NearestNeighbors()
    ann.fit(distance_i)
    for i in index_b:
        nearest = ann.kneighbors(distance_b[i],1, return_distance=False)
        fused_index.append(dict(
                index_base = i,
                index_inject = nearest[0][0],
                ))
        
    df_fused = pd.DataFrame(fused_index)
    df_ = pd.merge(df_fused, df_base, left_on='index_base', right_on='dummy_index')
    df_.drop('dummy_index', axis = 1, inplace=True)
    df_ = pd.merge(df_, df_inject, left_on ='index_inject', right_on='dummy_index')
    df_.drop(['dummy_index', 'index_base', 'index_inject'], axis = 1, inplace=True)
    return df_
    

def fuse_csv(csv_1, csv_2, distance_col = 'timestamp'):
    """
    Fuses two csv files
    %csv_1% - the path to the first csv file
    %csv_2% - the path to the second csv file
    %returns% - a data frame of the fused data
    """
    df_1 = pd.read_csv(csv_1)
    df_2 = pd.read_csv(csv_2)

    df_dense, df_sparse = compare_df(df_1, df_2)
    return fuse_data(df_dense, df_sparse, distance_col)



    

