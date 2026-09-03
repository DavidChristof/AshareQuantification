"""配置加载模块：读取 config/config.yaml 并提供全局访问。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# 项目根目录 = 本文件上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """简单的配置容器，支持 dict 风格访问 config['data']['universe']。"""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"Config 中没有 {key!r}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def resolve(self, path: str) -> Path:
        """把配置中的相对路径解析为基于项目根目录的绝对路径。"""
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict:
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    """加载 YAML 配置，默认读取 config/config.yaml。"""
    config_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 环境变量覆盖（可选，方便部署时调整）
    env_db = os.environ.get("QUANT_DB_PATH")
    if env_db:
        data["data"]["db_path"] = env_db

    # 数值健壮性：YAML 中科学计数法写法（如 1e-3）可能被解析为字符串，
    # 这里统一把模型参数中的数值字段转成 float，避免训练时报 TypeError。
    if "model" in data:
        for key in ("lr", "dropout"):
            if isinstance(data["model"].get(key), str):
                try:
                    data["model"][key] = float(data["model"][key])
                except ValueError:
                    pass

    return Config(data)
