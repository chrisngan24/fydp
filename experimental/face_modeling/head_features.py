"""
Generate features for classification
"""
import pandas as pd
import abc

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

def subtract_from_prev_val(df, col, step=1):
    """
    Subtract column value from the previous
    column value n steps away
    """
    return (df[col] - df.shift(periods=step)[col])

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
    active_features = set()
    for i, r in df.iterrows():
        w_start = i
        w_end   = min(i + 100, len(df)-1)
        row = r.to_dict()
        # drop the tail end of columns
        df_w = df.loc[w_start:w_end].reset_index(drop=True)
        for j in xrange(0,window):
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

    return pd.DataFrame(points), list(active_features)



def apply_feature_engineering(df, relevant_features = []):
    #return df
    df_b = df.copy()
    sub_frames = []
    # these are all "historic" systems
    ignore_columns = [c for c in df_b.columns.tolist() if c not in relevant_features]
    sub_frames.append(DeltaFeatureGenerator().generate_features(
            df, suffix='_6_steps', step=6,relevant_features=relevant_features, # eignore_columns,
            ))
    sub_frames.append(DeltaFeatureGenerator().generate_features(
            df, suffix='_1_steps', step=1, relevant_features=relevant_features,
            #ignore_columns=ignore_columns,
            ))
    df_new = pd.concat(sub_frames, axis=1)
    active_features = df_new.columns.values.tolist()
    for c in ignore_columns:
        if c in df.columns.values.tolist():
            df_new[c] = df[c]
    return df_new, active_features

