import os
import datetime as dt
from multiprocessing import Pool
from typing import Callable

import numpy as np
import pandas as pd
import sparkpickle
from tqdm import tqdm

from .diskfile import DiskFile
from .dataset import Dataset


def preprocess_data_template(row,
                             id: str,
                             feature_arrays: dict = {},
                             sort_by: str = 'event_time',
                             array_col: str = 'transaction',
                             one_level: bool = False
                             ):
    """
    Функция, которая принимает на вход транзакционный датафрейм для маппинга в rdd и сохраняет в
    sparkpickle
    Parameters:
        row: то, что будет приходить при вызове в rdd.map
        id: id (например, чека или клиента/завки)
        feature_arrays: dict, который переводит [имя фичи -> тип данных]
        sort_by: по чему сортировать
        array_col: как называется структура, в которой хранятся транзакции
        one_level: отдавать во flat виде или нет (лучше да, иначе потом для lifestream придется переводить во flat)
    """
    sorted_transactions = sorted(list(row[array_col]), key=lambda trans: trans[sort_by])

    feature_arrays = dict((col, np.array([trans[col] for trans in sorted_transactions], dtype=dtype)) if dtype
                          else (col, np.array([trans[col] for trans in sorted_transactions]))
                          for col, dtype in feature_arrays.items())

    if one_level:
        return list(
            {
                **feature_arrays,
                id: row[id]
            }.items()
        )
    else:
        return list(
            {
                'feature_arrays': feature_arrays,
                id: row[id]
            }.items()
        )


def read_pickle_template(file_path: str,
                         col_date: list,
                         array_col: str,
                         date_col: str,
                         id_col: str,
                         some_feature: str,
                         one_layer: bool = True
                         ):
    """
    Шаблон функции, которая считывает sparkpickle файл и сериализует его
    Должна быть обернута другой функцией (например, lambda)
    Parameters:
        file_path: путь к sparkpickle
        col_date: названия всех атрибутов, в которых был datetime формат (обычно не нужно)
        array_col: структура, в которой содержатся все фичи (обычно transactions)
        date_col: столбец времени подачи заявки (обычно не нужно)
        id_col: id
        some_feature: название любой фичи в transactions структуре
        one_layer: отдавать во flat виде или нет (лучше да, иначе потом для lifestream придется переводить во flat)
    """
    data = sparkpickle.load(open(file_path, 'rb'))
    if len(data) > 0:
        result = [dict(item) for item in data]
        for item in result:
            for col in col_date:
                item[array_col][col] = np.array([dt.datetime.strftime(date, '%Y-%m-%dT%H:%M:%S')
                                                 for date in item[array_col][col]])

        if one_layer:
            return [{f'{str(item[id_col])}': DiskFile.serialize_and_compress(item)}
                    for item in result if len(item[some_feature]) != 0]
        else:
            return [{f'{str(item[id_col])}': DiskFile.serialize_and_compress(item)}
                    for item in result if len(item[array_col][some_feature]) != 0]
    else:
        return []


def serialize_pickled(target_folder: str, file_paths: list, read_func: Callable):
    """
    Parameters:
        target_folder: папка, в которую будут записаны data.txt и index.txt
        file_paths: все пути к pickled файлам
        read_func: функция, принимающая 1 аргумент: путь к файлу
    """
    file_indices = dict()

    with open(os.path.join(target_folder, 'data.txt'), 'wb') as output:
        with Pool() as executors:
            for one_file in tqdm(executors.imap(read_func, file_paths)):
                if len(one_file) != 0:
                    for one_record in one_file:
                        for key, value in one_record.items():
                            file_indices[key] = (output.tell(), len(value))
                            output.write(value)

    with open(os.path.join(target_folder, 'index.txt'), 'wb') as output:
        output.write(DiskFile.serialize_and_compress(file_indices))


class DatasetsGenerator:
    """
    Класс, который создает подпоследовательности данных для каждого клиента
    """
    def __init__(self,
                 dataset: Dataset,
                 receipts_to_ids_mapper: dict,
                 id_col: str,
                 target_col: str,
                 min_receipts_num: int = 1,
                 ):
        """
        Parameters:
            dataset: датасет с данными
            receipts_to_ids_mapper: маппер из номеров id чеков в id клиентов
            id_col: id клиента
            target_col: название целевой переменной в dataset
            min_receipts_num: минимальный номер чека в данных
        """
        self.dataset = dataset
        self.receipts_to_ids_mapper = receipts_to_ids_mapper
        self.id_col = id_col
        self.target_col = target_col
        self.min_receipts_num = min_receipts_num
        self.id_map = f'{self.id_col}_map'
        self.features = None
        self.exception_idx = 1
        self.next_dataset_start_receipt_id = 1

    def __call__(self, embeddings: np.ndarray):
        return self._make_dataset(embeddings)

    def _make_dataset(self, embeddings: np.ndarray):
        df = pd.DataFrame(embeddings)
        df.columns = [str(col) for col in df.columns]
        if self.features is None:
            self.features = df.columns.values.copy()
        df[self.id_map] = np.arange(self.next_dataset_start_receipt_id,
                                    self.next_dataset_start_receipt_id + df.shape[0])

        self.next_dataset_start_receipt_id = self.next_dataset_start_receipt_id + df.shape[0]

        df[self.id_col] = df[self.id_map].map(self.receipts_to_ids_mapper)
        current_dataset = []
        for name, group in tqdm(df.groupby(self.id_col, as_index=False)):
            if len(group) <= self.min_receipts_num:
                continue
            grp = group[self.features].T
            dct = dict(zip(grp.index, grp.values))
            target = self.dataset[group[self.id_map].values[-1]][self.target_col][-1]
            current_dataset.append(({self.id_col: name, **dct}, target))
        return current_dataset

