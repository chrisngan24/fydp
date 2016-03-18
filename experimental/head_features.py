"""
Generate features for classification
"""
import pandas as pd
import abc

import numpy as np

class FeatureFactor:
    """
    Generic class for feature generations
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass

    def get_active_columns(self, df, ignore_columns):
        """
        Assumes active columns are all columns in DF
        except those that that are listed in ignore_columns
        """
        cols = df.columns.values.tolist()
        print cols
        for c in ignore_columns:
            if c in cols:
                cols.remove(c)
        return cols

    @abc.abstractmethod
    def generate_features(self, df, relevant_columns, ignore_columns=[],**kwargs):
        """
        Some computation that does 
        feature generation
        """
        return df

def subtract_from_prev_val(df, col, step=1, step_col = None):
    """
    Subtract column value from the previous
    column value n steps away. 
    Optional param to divide by difference in column
    (like time)
    """
    diff_vector =  (df[col] - df.shift(periods=step)[col])
    if step_col == None:
        return diff_vector
    else:
        step_vector =  (df[step_col] - df.shift(periods=step)[step_col])
        step_vector.fillna(0, inplace=True)
        return diff_vector / step_vector

class DeltaTimeFeatureGenerator(FeatureFactor):
    """
    Take features from n steps away to compute 
    a velocity feature vector. Uses the data points from
    the past.
    """
    def generate_features(self, df, suffix = '', step=1, relevant_features=[], ignore_columns=[]):
        """
        Generate the features, returns a new data frame of all 
        transformed features (same length as input)
        :param df: - input data frame
        :param suffix: - the ending of the new column, default is change nothing
                         to column name
        :param step: - delta from how many index periods away
        :param ignore_columns: - what are the columns to ignore
        """
        # cols = self.get_active_columns(df, ignore_columns)
        cols = relevant_features
        deltas = {}
        for c in cols:
            deltas['%s%s'% (c, suffix)] = \
                    subtract_from_prev_val(df, c, step=step, step_col='time')
        df_new = pd.DataFrame(deltas)
        return df_new

class BooleanSummer(FeatureFactor):
    """
    Take features from 1, 2,... n steps away to compute 
    the number of trues for a given boolean column
    """
    def generate_features(self, df, suffix = '', step=1, relevant_features=[], ignore_columns=[]):
        """
        Generate the features, returns a new data frame of all 
        transformed features (same length as input)
        :param df: - input data frame
        :param suffix: - the ending of the new column, default is change nothing
                         to column name
        :param step: - delta from how many index periods away
        :param ignore_columns: - what are the columns to ignore
        """
        # cols = self.get_active_columns(df, ignore_columns)
        cols = relevant_features
        deltas = {}
        for c in cols:
            col_name = '%s%s' % (c, suffix)
            de = pd.DataFrame({col_name : [0]*len(df)})
            for i in xrange(step):
                a = df.shift(periods=i)[c].fillna(0)
                de[col_name] = de[col_name] + list(a)
                
            deltas[col_name] = \
                    de[col_name] / step
        df_new = pd.DataFrame(deltas)
        return df_new




class DeltaFeatureGenerator(FeatureFactor):
    """
    Take features from n steps away to compute 
    a velocity feature vector. Uses the data points from
    the past.
    """
    def generate_features(self, df, suffix = '', step=1, relevant_features=[], ignore_columns=[]):
        """
        Generate the features, returns a new data frame of all 
        transformed features (same length as input)
        :param df: - input data frame
        :param suffix: - the ending of the new column, default is change nothing
                         to column name
        :param step: - delta from how many index periods away
        :param ignore_columns: - what are the columns to ignore
        """
        # cols = self.get_active_columns(df, ignore_columns)
        cols = relevant_features
        deltas = {}
        for c in cols:
            deltas['%s%s'% (c, suffix)] = \
                    subtract_from_prev_val(df, c, step=step)
        df_new = pd.DataFrame(deltas)
        return df_new

def generate_windows(df, window=10, relevant_features = []):
    """
    Take the future points - up to a specific window size -
    and add it to the current row as a set of features
    """
    points = []
    cols = df.columns.values.tolist()   
    active_features = set(relevant_features)
    for i, r in df.iterrows():
        w_start = i
        w_end   = min(i + 100, len(df)-1)
        row = r.to_dict()
        # drop the tail end of columns
        df_w = df.loc[w_start:w_end].reset_index(drop=True)
        for j in xrange(1,window):
            if j < len(df_w):
                window_row = df_w.loc[j].to_dict()
            else:
                window_row = None
            for c in cols:
                if c in relevant_features:
                    name = '%s_%s' % (c, j)
                    row[name] = window_row[c] if window_row != None else None
                    if not name in active_features:
                        active_features.add(name)
        points.append(row)
    df = pd.DataFrame(points).fillna(0)

    return df, list(active_features)


# OG one
# def apply_feature_engineering(df, relevant_features = [], window_size=10):
#     #return df
#     df_b = df.copy()
#     sub_frames = []
#     # these are all "historic" systems
#     ignore_columns = [c for c in df_b.columns.tolist() if c not in relevant_features]
#     sub_frames.append(DeltaFeatureGenerator().generate_features(
#             df, suffix='_6_steps', step=6,relevant_features=relevant_features, # eignore_columns,
#             ))
# 
#     sub_frames.append(DeltaFeatureGenerator().generate_features(
#             df, suffix='_1_steps', step=1, relevant_features=relevant_features,
#             #ignore_columns=ignore_columns,
#             ))
#     df_new = pd.concat(sub_frames, axis=1)
#     #df_new['pos_delta_x'] = df_new['noseX_1_steps'] > 0 
#     active_features = df_new.columns.values.tolist()
#     for c in df.columns.values.tolist():
#         if not c in df_new.columns.values.tolist():
#             df_new[c] = df[c]
#     df_w, active_features = generate_windows(df_new, window=window_size, relevant_features=active_features) 
#     return df_w, active_features


from scipy.special import expit
# def apply_feature_engineering(df, relevant_features = [], window_size =10):
#     ## Much more control of features
#     df_b = df.copy()
#     sub_frames = []
#     # these are all "historic" systems
#     ignore_columns = [c for c in df_b.columns.tolist() if c not in relevant_features]
#     delta_columns = ['noseX', 'noseY', 'faceLeft', 'faceRight']
#     
#     #sub_frames.append(DeltaTimeFeatureGenerator().generate_features(
#     #        df, suffix='_6_steps', step=6,relevant_features=delta_columns, # eignore_columns,
#     #        ))
#     sub_frames.append(DeltaTimeFeatureGenerator().generate_features(
#             df, suffix='_2_steps', step=2,relevant_features=delta_columns, # eignore_columns,
#             ))
#     sub_frames.append(DeltaTimeFeatureGenerator().generate_features(
#             df, suffix='_3_steps', step=3,relevant_features=delta_columns, # eignore_columns,
#             ))
#     sub_frames.append(DeltaTimeFeatureGenerator().generate_features(
#             df, suffix='_1_steps', step=1, relevant_features=delta_columns,
#             #ignore_columns=ignore_columns,
#             ))
#     df_new = pd.concat(sub_frames, axis=1)
#     for c in df_new.columns.tolist():
#         if c.find('face') == 0:
#             df_new[c] = expit(df_new[c])
#     df_new['pos_delta_x'] = df_new['noseX_1_steps'] > 0 
#     df_new['isFrontFace'] = df['isFrontFace']
#     active_features = df_new.columns.values.tolist()
# 
#     for c in df.columns.values.tolist():
#         if not c in df_new.columns.values.tolist():
#             # to scale 
#             df_new[c] = df[c]
#     df_w = df_new
#     df_w, active_features = generate_windows(df_new, window=window_size, relevant_features=active_features)
#     return df_w, active_features


def apply_feature_engineering(df, relevant_features = [], window_size =10):
    ## Much more control of features
    df_b = df.copy()
    sub_frames = []
    # these are all "historic" systems
    ignore_columns = [c for c in df_b.columns.tolist() if c not in relevant_features]
    delta_columns = ['noseX', 'noseY', 'faceLeft', 'faceRight']
    
#     sub_frames.append(DeltaTimeFeatureGenerator().generate_features(
#             df, suffix='_6_steps', step=6,relevant_features=delta_columns, # eignore_columns,
#             ))
    sub_frames.append(DeltaTimeFeatureGenerator().generate_features(
            df, suffix='_2_steps', step=2,relevant_features=delta_columns, # eignore_columns,
            ))
    sub_frames.append(DeltaTimeFeatureGenerator().generate_features(
            df, suffix='_3_steps', step=3,relevant_features=delta_columns, # eignore_columns,
            ))
    sub_frames.append(DeltaTimeFeatureGenerator().generate_features(
            df, suffix='_1_steps', step=1, relevant_features=delta_columns,
            #ignore_columns=ignore_columns,
            ))
    df_new = pd.concat(sub_frames, axis=1)
    for c in df_new.columns.tolist():
        if c.find('face') == 0:
            # to bound the data
            df_new[c] = expit(df_new[c])
    active_features = df_new.columns.values.tolist()
    df_new['pos_delta_x'] = df_new['noseX_1_steps'] > 0 
    for c in delta_columns:
        rel_cols = []
        for ac in active_features:
            if ac.find(c) == 0:
                rel_cols.append(ac)
        df_new['%s_median' % c] = df_new[rel_cols].apply(np.median, axis=1)
        df_new.drop(rel_cols, inplace=True, axis=1)

    df_new['isFrontFace'] = df['isFrontFace']
    active_features = df_new.columns.values.tolist()

    for c in df.columns.values.tolist():
        if not c in df_new.columns.values.tolist():
            # to scale 
            df_new[c] = df[c]
    df_w = df_new
    df_w, active_features = generate_windows(df_new, window=window_size, relevant_features=active_features)
    return df_w, active_features



def apply_feature_engineering(df, relevant_features = [], window_size=10):
    #return df
    df_b = df.copy()
    sub_frames = []
    # these are all "historic" systems
    delta_cols = ['noseX', 'noseY']
    ignore_columns = [c for c in df_b.columns.tolist() if c not in relevant_features]
    sub_frames.append(DeltaFeatureGenerator().generate_features(
            df, suffix='_6_steps', step=6,relevant_features=delta_cols, # eignore_columns,
            ))
    sub_frames.append(DeltaFeatureGenerator().generate_features(
            df, suffix='_4_steps', step=4,relevant_features=delta_cols, # eignore_columns,
            ))
    sub_frames.append(DeltaFeatureGenerator().generate_features(
            df, suffix='_3_steps', step=3,relevant_features=delta_cols, # eignore_columns,
            ))

    sub_frames.append(DeltaFeatureGenerator().generate_features(
            df, suffix='_1_steps', step=1, relevant_features=delta_cols,
            #ignore_columns=ignore_columns,
            ))
    sub_frames.append(BooleanSummer().generate_features(
            df, suffix='_face_counter', step=4, relevant_features=['isFrontFace'],
            #ignore_columns=ignore_columns,
            ))
    df_new = pd.concat(sub_frames, axis=1)
    df_new['pos_delta_x'] = df_new['noseX_1_steps'] > 0 
    df_new['isFrontFace'] = df['isFrontFace']
    active_features = df_new.columns.values.tolist()
    for c in df.columns.values.tolist():
        if not c in df_new.columns.values.tolist():
            df_new[c] = df[c]
    df_w, active_features = generate_windows(df_new, window=window_size, relevant_features=active_features) 
    return df_w, active_features

