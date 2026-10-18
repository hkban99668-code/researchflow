"""T-030 BM25：手算小例子得分一致；中文二元组与英文词都能命中（A4）。"""

from __future__ import annotations

import math

import pytest

from researchflow.tools.bm25 import BM25Index, tokenize


def test_tokenize_rules():
    tokens = tokenize("Missing Data Imputation 缺失值插补 a b")
    assert "missing" in tokens and "data" in tokens, "英文小写词"
    assert "a" not in tokens and "b" not in tokens, "长度 1 的英文词去掉"
    assert "缺失" in tokens and "失值" in tokens and "插补" in tokens, "中文字二元组"
    assert "缺失值" not in tokens


def test_bm25_hand_computed_scores():
    # 手算：docs = ["cat cat dog", "cat bird"]，query = "cat dog"
    # N=2, avg_len=2.5, k1=1.5, b=0.75
    # idf(cat)=ln(1+0.2/2.5)?  展开：ln(1+(2-2+0.5)/(2+0.5))=ln(1.2)
    # idf(dog)=ln(1+(2-1+0.5)/(1+0.5))=ln(2)
    # doc0: |d|=3, norm=1.5*(0.25+0.75*3/2.5)=1.725
    #   cat: ln1.2 * 2*2.5/(2+1.725)；dog: ln2 * 2.5/(1+1.725)
    # doc1: |d|=2, norm=1.275；cat: ln1.2 * 2.5/(1+1.275)
    index = BM25Index.build(["cat cat dog", "cat bird"])
    idf_cat = math.log(1.2)
    idf_dog = math.log(2.0)
    doc0 = idf_cat * 2 * 2.5 / (2 + 1.725) + idf_dog * 2.5 / (1 + 1.725)
    doc1 = idf_cat * 2.5 / (1 + 1.275)
    ranked = index.rank("cat dog")
    assert ranked[0] == (0, pytest.approx(doc0))
    assert ranked[1] == (1, pytest.approx(doc1))
    assert ranked[0][1] > ranked[1][1], "同时命中两个词的文档排前"


def test_bm25_chinese_bigram_hits():
    index = BM25Index.build(["缺失值插补方法综述", "回归系数估计方法"])
    ranked = index.rank("插补")
    assert ranked[0][0] == 0, "查询二元组'插补'命中含'插补'的文档"
    assert ranked[0][1] > 0
    assert index.rank("回归")[0][0] == 1


def test_bm25_rank_k_and_stable_ties():
    index = BM25Index.build(["dog", "dog", "cat"])
    ranked = index.rank("dog", k=2)
    assert len(ranked) == 2
    assert all(doc_index in (0, 1) for doc_index, _ in ranked), "同分按文档序号稳定排序"
    assert ranked[0][1] >= ranked[1][1]
