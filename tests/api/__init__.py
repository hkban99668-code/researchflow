"""API 验收测试包。

必须是包（有 `__init__.py`）：pytest 的 prepend 导入模式会把非包目录插进
sys.path，tests/api/conftest.py 与 tests/conftest.py 撞 `conftest` 模块名，
破坏 e2e 的 `from conftest import thread_config`。打包后本目录的 conftest
以 `api.conftest` 导入，tests/api 也不再进 sys.path。
"""
