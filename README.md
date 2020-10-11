# PLEML

Introduction
--

Label Enhancement (LE) aims to recover the hidden label distribution value from the logical labels of the data sets.

Publication
--

Code accompanying paper **Privileged Label Enhancement with Multi-Label Learning**. IJCAI 2020.  https://www.ijcai.org/Proceedings/2020/329

DataSet
--

The group of PALM provides some LDL data sets. http://palm.seu.edu.cn/xgeng/LDL/index.htm

How to use
--

1) **MLL_LowRANK** : the first part of PLEML, run rum.m and the result is stored in the "result_mll" folder.

2) **LE_RSVM**+ : the second part of PLEML, rum rsvm+.py and the folder "result_mll" is the results of MLL_LowRANK.

## Environment

Ubuntu 18.04

Matlab R2016a

PyCharm 2018

Intel® Core™ i5-6500 CPU @ 3.20GHz × 4