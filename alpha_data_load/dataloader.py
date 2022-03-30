from .dataset import Dataset

def get_easy_batch_transaction(data, dataloader, preparation=lambda x: x):
    """
    Такие функции, которые передаются через DataLoader, можно писать, чтобы на сервере
    предобрабатывать данные каким-то несложным образом
    """
    return preparation(data)


class DataLoaderIterator:
    """
    Iterator for DataLoader class
    """
    def __init__(self,
                 dataloader,
                 batch_size,
                 get_batch_func=get_easy_batch_transaction
                 ):
        self._dataloader = dataloader
        self.batch_size = batch_size
        self.get_batch_func = get_batch_func
        self.current_idx = 0

    def __next__(self):
        if self.current_idx >= len(self._dataloader.dataset):
            raise StopIteration
        batch = self._create_batch()
        self.current_idx += self.batch_size
        return batch

    def _create_batch(self):
        #момент загрузки данных в оперативку
        data = self._dataloader.dataset[self.current_idx : self.current_idx + self.batch_size]
        return self.get_batch_func(data=data, dataloader=self._dataloader)


class DataLoader:
    """
    Создает объект, по которому можно итерироваться для частичной подгрузки данных и дообучения
    """
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 get_batch_func=get_easy_batch_transaction,
                 shuffle: bool = False
                 ):
        self.dataset = dataset
        self.get_batch_func = get_batch_func
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            dataset.shuffle()

    def __iter__(self):
        return DataLoaderIterator(self, self.batch_size, get_batch_func=self.get_batch_func)