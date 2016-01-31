import abc

class SignalClassifier:
    """
    Parent class to classifies the signals

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, df, signal_indices):
        """
        :param df: - the entire time series signal
        :param signal_indices: - a list of tuples that 
                                 represent an event
                                 
                    (start_index, end_index, event_name)
        """
        self.df = df.copy()
        self.signal_indices = signal_indices

    @abc.abstractmethod
    def classify_signals(self):
        """
        initiate the classify function
        """
        events = [
            dict(
                event_type='', 
                start_index=0,
                end_index=1,
                classification='',
                )
            ]
        return events

