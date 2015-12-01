"""
Generate features for classification
"""
import pandas as pd
import abc

class FeatureFactor:
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass

    def get_active_columns(self, df, ignore_columns):
        cols = df.columns.values.tolist()
        for c in ignore_columns:
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
    return (df[col] - df.shift(periods=step)[col])

class DeltaFeatureGenerator(FeatureFactor):
    def generate_features(self, df, suffix = '', step=1, ignore_columns=[]):
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
        df_new[c] = df[c]
    return df_new

