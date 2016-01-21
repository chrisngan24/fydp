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
    def generate_features(self, df, ignore_columns=[],**kwargs):
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
    a velocity feature vector
    """
    def generate_features(self, df, suffix = '', step=1, ignore_columns=[]):
        """
        Generate the features, returns a new data frame of all 
        transformed features (same length as input)
        :param df: - input data frame
        :param suffix: - the ending of the new column, default is change nothing
                         to column name
        :param step: - delta from how many index periods away
        :param ignore_columns: - what are the columns to ignore
        """
        cols = self.get_active_columns(df, ignore_columns)
        deltas = {}
        for c in cols:
            deltas['%s%s'% (c, suffix)] = \
                    subtract_from_prev_val(df, c, step=step)
        df_new = pd.DataFrame(deltas)
        return df_new



def apply_feature_engineering(df, ignore_columns = []):
    #return df
    df_b = df.copy()
    sub_frames = []
    sub_frames.append(DeltaFeatureGenerator().generate_features(
            df, suffix='_6_steps', step=6, ignore_columns=ignore_columns,
            ))
    sub_frames.append(DeltaFeatureGenerator().generate_features(
            df, suffix='_1_steps', step=1, ignore_columns=ignore_columns,
            ))
    df_new = pd.concat(sub_frames, axis=1)
    for c in ignore_columns:
        if c in df.columns.values.tolist():
            df_new[c] = df[c]
    return df_new

