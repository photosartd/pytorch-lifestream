import io
import pickle as pkl

import numpy as np
from lz4 import frame


class DiskFile(object):
    """
    Просто твой класс для считывания объектов с диска
    """
    def __init__(self, file_path: str, index_path: str):
        self.file_path = file_path
        with open(index_path, 'rb') as inp:
            self.index = self.decompress_and_deserialize(inp.read()).copy()

    def __contains__(self, idx):
        return idx in self.index

    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            pos = []
            bytes_list = []
            result = []
            for i in idx:
                position, n_bytes = self.index[i]
                pos.append(position)
                bytes_list.append(n_bytes)
            with open(self.file_path, 'rb') as inp:
                for position, n_bytes in zip(pos, bytes_list):
                    inp.seek(position, 0)
                    data = inp.read(n_bytes)
                    result.append(self.decompress_and_deserialize(data))
            return result
        else:
            if idx not in self:
                raise KeyError(f'No item with such index: {idx}')
            position, n_bytes = self.index[idx]
            """
            TODO: нужно переопределить слайсы, чтобы открывать файл только 1 раз 
            для больших батчей и читать последовательно
            """
            with open(self.file_path, 'rb') as inp:
                inp.seek(position, 0)
                data = inp.read(n_bytes)
        return self.decompress_and_deserialize(data)


    @staticmethod
    def decompress_and_deserialize(data):
        return pkl.loads(frame.decompress(data))

    @staticmethod
    def serialize_and_compress(data):
        pickle_buffer = io.BytesIO()
        pkl.dump(data, pickle_buffer, protocol=4)
        compressed = frame.compress(pickle_buffer.getbuffer(), content_checksum=True)
        return compressed