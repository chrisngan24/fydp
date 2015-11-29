import abc

class EventAnnotator:
    """
    Abstract class for annotatinv events in
    the index time series data
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def annotate_events(self, df):
        """
        Annotate events in the
        time series df
        :return: - events = {
                    event_name : [index_0, index_1, ...]
                }
        """
        events = {}
        return events


