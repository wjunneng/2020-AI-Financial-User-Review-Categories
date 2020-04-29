# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys
import pandas as pd

os.chdir(sys.path[0])

pd.set_option('display.max_columns', 1000)

pd.set_option('display.width', 1000)

pd.set_option('display.max_colwidth', 1000)


class EDA(object):
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(self.path)

    def main(self):
        """
        主方法
        :return:
        """

        # ############################################ 未去重 ############################################
        """
                          label
            count  18181.000000
            mean       2.680656
            std        2.009218
            min        0.000000
            25%        1.000000
            50%        2.000000
            75%        4.000000
            max       10.000000
        """
        print(self.data.describe())
        print('\n')

        print('---------------------------------------------')
        """
            <class 'pandas.core.frame.DataFrame'>
            RangeIndex: 18181 entries, 0 to 18180
            Data columns (total 2 columns):
            text     18181 non-null object
            label    18181 non-null int64
            dtypes: int64(1), object(1)
            memory usage: 284.2+ KB
            None           
        """
        print(self.data.info())
        print('\n')

        print('---------------------------------------------')
        # ############################################# 去重 #############################################
        self.data = self.data.drop_duplicates()

        self.data.reset_index(drop=True, inplace=True)

        """
                          label
            count  18115.000000
            mean       2.676622
            std        2.011418
            min        0.000000
            25%        1.000000
            50%        2.000000
            75%        4.000000
            max       10.000000         
        """
        print(self.data.describe())
        print('\n')

        print('---------------------------------------------')
        """
           <class 'pandas.core.frame.DataFrame'>
            RangeIndex: 18115 entries, 0 to 18114
            Data columns (total 2 columns):
            text     18115 non-null object
            label    18115 non-null int64
            dtypes: int64(1), object(1)
            memory usage: 283.2+ KB
            None
        """
        print(self.data.info())
        print('\n')

        print('---------------------------------------------')
        """
            label   cnt
        0       0  1135
        1       1  6129
        2       2  2445
        3       3  3153
        4       4  1968
        5       5   780
        6       6  1964
        7       7   129
        8       8   199
        9       9   172
        10     10    41        
        """
        print(self.data.groupby(['label'], as_index=False)['label'].agg({'cnt': 'count'}))


if __name__ == '__main__':
    path = '../../data/input/train.csv'

    eda = EDA(path=path)

    eda.main()
