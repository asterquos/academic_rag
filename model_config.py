"""
model_config.py
统一的模型配置和缓存管理
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

# 全局模型缓存目录
MODEL_CACHE_DIR = Path(__file__).parent / "models" / "embeddings"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 设置环境变量（确保所有模型库都使用这个目录）
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR.parent)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_HUB_CACHE'] = str(MODEL_CACHE_DIR)


def get_embedding_model_config(
        model_type: str = 'bge-large-zh',
        use_fp16: bool = True,
        batch_size: int = 64,
        device: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取嵌入模型的统一配置

    Args:
        model_type: 模型类型
        use_fp16: 是否使用半精度
        batch_size: 批处理大小
        device: 设备类型

    Returns:
        模型配置字典
    """
    return {
        'model_type': model_type,
        'cache_folder': str(MODEL_CACHE_DIR),
        'use_fp16': use_fp16,
        'batch_size': batch_size,
        'device': device
    }


def get_model_info():
    """获取已缓存的模型信息"""
    info = {
        'cache_directory': str(MODEL_CACHE_DIR),
        'cached_models': [],
        'total_size': 0
    }

    if MODEL_CACHE_DIR.exists():
        for model_dir in MODEL_CACHE_DIR.glob("*"):
            if model_dir.is_dir():
                size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                info['cached_models'].append({
                    'name': model_dir.name,
                    'size_mb': size / 1024 / 1024
                })
                info['total_size'] += size

    info['total_size_mb'] = info['total_size'] / 1024 / 1024
    return info


def set_offline_mode(offline: bool = True):
    """设置离线模式（避免检查更新）"""
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    else:
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)


# 在导入时就设置环境变量
print(f"模型缓存目录: {MODEL_CACHE_DIR}")