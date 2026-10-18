"""检索评测：候选池生成（pooling）、人工标注表与 Recall@k / MRR 指标。

三个文件：

- ``topics.yaml``：课题清单（本卡只放 3 条示例，正式 20 条由 gu 填写）；
- ``build_pool.py``：候选池生成与标注表（``pool.json`` + ``labels_todo.csv``）；
- ``score.py``：读回 gu 填好的标注，算各方法的 Recall@k / Precision@k / MRR，
  并报告新运行中池外文献的"未标注"计数。

候选池在标注前冻结：``pool.json`` 记录 OpenAlex 缓存目录的 checksum，之后改了
检索方式再测，新出现但不在池里的文献记为"未标注"，不悄悄当"不相关"。
"""
