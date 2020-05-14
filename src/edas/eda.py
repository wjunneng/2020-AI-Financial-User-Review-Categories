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
        self.data = pd.read_csv(self.path, header=None, names=['id', 'label'])

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
        0       0  1135 消费贷款
        1       1  6129 抵押
        2       2  2445 信用卡
        3       3  3153 债务催收
        4       4  1968 信用报告
        5       5   780 学生贷款
        6       6  1964 银行账户服务
        7       7   129 短期小额贷款
        8       8   199 汇款
        9       9   172 预付卡
        10     10    41 其他金融服务       
        
        消费贷款/学生贷款短期小额贷款
        信用卡/信用报告预付卡/汇款
        抵押/债务催收
        银行账户服务/其他金融服务
        
        """
        print(self.data.groupby(['label'], as_index=False)['label'].agg({'cnt': 'count'}))


if __name__ == '__main__':
    path = '../../data/input/train.csv'
    # path = '../../data/output/keys_90_21.csv'
    # path = '../../data/output/keys_89_81.csv'

    eda = EDA(path=path)

    eda.main()
