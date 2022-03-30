"""
Здесь лежит класс для препроцессинга таблички в транзакционный вид
(основано на задаче чеков в Sber PreLab)
Должно выполняться на кластере до переноса на сервер
"""
from typing import List

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *

from .base import DataPreprocessor


class PysparkDataPreprocessor(DataPreprocessor):
    """
    Использование:

    df = ...
    preprocessor = PysparkDataPreprocessor(...)
    df_transactions = preprocessor.fit_transform(df)
    """
    def __init__(self, col_id: str, cols_event_time: str, cols_category: List[str], cols_log_norm: List[str],
                 target_col: str = None, time_transformation: str = 'hours_from_min', print_dataset_info: bool = False,
                 null_fill_value: str = '_', null_fill_numeric_value: float = 0.0):
        """
        Parameters:
            col_id: id, по которому будет происходить группировка
            cols_event_time: название атрибута, который представляет время
            cols_category: категориальные фичи
            cols_log_norm: numerical фичи (будут отскейлены логнормированием)
            target_col: атрибут таргета; если None, то в транзакционной табличке его не будет;
             никак не предобрабатывается
            time_transformation: пока доступен только 'hours_from_min', который отсчитывает часы от минимального часа
            в датасете; опции можно посмотреть в pandas_preprocessor и реализовать
            print_dataset_info: печатать ли отладочную информацию
            null_fill_value: значение, которым заполняются категориальные переменные
            null_fill_numeric_value: значение, которым заполняются все numerical nulls
        """
        super().__init__(col_id, cols_event_time, cols_category, cols_log_norm)
        self.EVENT_TIME = 'event_time'
        self.time_transformation = time_transformation
        self.print_dataset_info = print_dataset_info
        self.null_fill_value = null_fill_value
        self.null_fill_numeric_value = null_fill_numeric_value
        self.target_col = target_col
        self.enum_col = 'enum'

    def fit(self, df_: pyspark.sql.DataFrame, **params):
        """
        Parameters:
            df_: pyspark.DataFrame with flat data
        Returns:
            self : object
            Fitted preprocessor
        """
        self._reset()
        df = df_.alias('df')

        """
        Создаем маппинги для категориальных переменных
        """
        for column in self.cols_category:
            ps_col = df.select(F.col(column).cast(StringType()))
            mapping = ps_col.fillna(self.null_fill_value,
                                    subset=[column]
                                    ) \
                .groupBy(column) \
                .count() \
                .select(column) \
                .withColumn(self.enum_col,
                            F.dense_rank() \
                            .over(Window.partitionBy().orderBy(column))
                            )
            self.cols_category_mapping[column] = mapping

        return self

    def transform(self, df: pyspark.sql.DataFrame, **params):
        self.check_is_fitted()
        for column in self.cols_log_norm:
            df = df.withColumn(column, F.col(column).cast(DoubleType())) \
                .fillna(self.null_fill_numeric_value, subset=[column])
        df.fillna(self.null_fill_numeric_value, subset=self.cols_log_norm)
        if self.target_col:
            assert self.target_col in df.columns, 'Target column was not in df.columns'
            new_name = f'target_{self.target_col}'
            df = df.withColumn(new_name, F.col(self.target_col))
            self.target_col = new_name

        """
        TODO:
        Здесь нужно доделать получение временных фичей (ночь, час и тд)
        """

        # event time mapping
        if self.time_transformation == 'hours_from_min':
            df = self._td_hours(df, self.cols_event_time)
            df = df.fillna(-1, subset=[self.cols_event_time])
        else:
            raise NotImplementedError('This time transformation was not implemented yet')

        for column in self.cols_category_mapping:
            if column not in self.cols_category_mapping:
                raise KeyError(f'column {column} isn"t in fitted category columns')
            df = df.withColumn(column, F.col(column).cast(StringType())) \
                .fillna(self.null_fill_value, subset=[column]) \
                .join(self.cols_category_mapping[column],
                      on=[column]
                      ) \
                .drop(column) \
                .withColumnRenamed(self.enum_col, column)

        for column in self.cols_log_norm:
            df = df.withColumn(column, F.log1p(F.abs(F.col(column))) * F.signum(F.col(column)))
            df = df.withColumn(column, F.col(column) / F.max(F.abs(F.col(column))).over(Window.orderBy()))

        # column filter
        used_columns = [col for col in df.columns
                        if col in self.cols_category + self.cols_log_norm + [self.EVENT_TIME]]

        if self.target_col:
            used_columns += [self.target_col]

        df.sort([self.col_id, self.EVENT_TIME])
        if self.print_dataset_info:
            nans = df.select([F.colunt(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
                              for c in used_columns]).collect()
            print('NaN values in each columns:')
            print(nans)

        df_trans = self._to_transactions_table(df, features=used_columns, app_id=self.col_id)

        return df_trans

    def get_category_sizes(self):
        """
        Возвращает маппинг категориальная_фича -> размер словаря категорий
        Нужно, чтобы подавать в seq_encoder object
        """
        return {column: df.count() + 1 for column, df in self.cols_category_mapping.items()}

    def _td_hours(self, df, col_event_time):
        df = df.withColumn(self.EVENT_TIME, (F.unix_timestamp(F.col(col_event_time)) -
                                             F.unix_timestamp(F.min(col_event_time).over(Window.orderBy()))) / 3600)
        return df

    @staticmethod
    def _to_transactions_table(df: pyspark.sql.DataFrame,
                               features: list,
                               app_id: str,
                               others: list = [],
                               array_col: str = 'transactions'
                               ):
        """
        Функция, которая переводит таблицу в транзакционный вид
        Parameters:
            df: отсортированный по [APP_ID, TRANS_DATE] датафрейм
            features: фичи, которые должны лежать в массиве массивов
            app_id: название атрибута id заявки
            others: список фичей, который должны лежать рядом с id (выше массивов)
            return pyspark.sql.DataFrame
        """
        ELEMENT = 'element'
        aggs = [F.max(column).alias(column) for column in others]
        df_trans = df.select(app_id, *others, F.struct(*features).alias(ELEMENT)) \
            .groupBy([app_id])\
            .agg(*aggs, F.collect_list(ELEMENT).alias(array_col))
        return df_trans
