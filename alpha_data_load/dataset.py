import numpy as np

from .diskfile import DiskFile


class Dataset(object):
    """
    Класс, который абстрагирует DiskFile от id и позволяет обращаться по индексу, а также с помощью слайсов
    """

    def __init__(self,
                 data: DiskFile,
                 app_id: str,
                 app_date: str,
                 features: str,
                 some_feature: str,
                 sort_indexes=True
                 ):
        self.data = data
        self.app_id = app_id
        self.app_date = app_date
        self.features = features
        self.some_feature = some_feature
        self.indexes = list(self.data.index.keys())
        if sort_indexes:
            self.indexes.sort(key=int)
        self.indexes = np.array(self.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if isinstance(idx.step, int) and idx.step != 1:
                raise AttributeError('Step must be 1 in slice')
            if not isinstance(idx.start, int):
                start = 0
            elif idx.start >= len(self.indexes):
                raise KeyError(f'No such indexes: {idx.start} - {idx.stop}')
            else:
                start = idx.start
            if idx.stop > len(self.indexes):
                stop = len(self.indexes)
            else:
                stop = idx.stop
            return list(map(self.__getitem__, range(start, stop)))
        if idx > len(self.indexes):
            raise KeyError(f'Index {idx} not in self.indexes')
        return self.data[self.indexes[idx]]

    def shuffle(self):
        self.indexes = np.random.shuffle(self.indexes)
