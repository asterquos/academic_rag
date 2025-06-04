"""
clean_data.py
数据清理脚本 - 用于过滤和改进数据质量
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_art_design_data(
        input_file: str = "data/processed/processed_data.parquet",
        output_file: str = "data/processed/processed_data_clean.parquet",
        min_text_length: int = 50,
        fill_missing_issue: bool = True,
        remove_duplicates: bool = True,
        remove_empty_titles: bool = True
):
    """
    清理艺术设计数据

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        min_text_length: 最小文本长度
        fill_missing_issue: 是否填充缺失的刊号
        remove_duplicates: 是否移除重复内容
        remove_empty_titles: 是否移除空标题
    """
    logger.info(f"加载数据: {input_file}")
    df = pd.read_parquet(input_file)
    original_count = len(df)
    logger.info(f"原始数据: {original_count} 条")

    # 1. 填充缺失的刊号（如果需要）
    if fill_missing_issue and '刊号' in df.columns:
        before_fill = df['刊号'].isna().sum()
        df['刊号'] = df['刊号'].ffill().bfill()
        after_fill = df['刊号'].isna().sum()
        logger.info(f"填充刊号: {before_fill} -> {after_fill} 个空值")

    # 2. 过滤短文本（可能是图片标题等）
    if '全文长度' in df.columns:
        short_text_count = (df['全文长度'] < min_text_length).sum()
        df = df[df['全文长度'] >= min_text_length]
        logger.info(f"移除短文本（<{min_text_length}字符）: {short_text_count} 条")

    # 3. 移除空标题
    if remove_empty_titles and '文章名称+副标题' in df.columns:
        empty_title_count = (df['文章名称+副标题'].str.strip() == '').sum()
        df = df[df['文章名称+副标题'].str.strip() != '']
        logger.info(f"移除空标题: {empty_title_count} 条")

    # 4. 移除重复内容
    if remove_duplicates and '全文' in df.columns:
        before_dedup = len(df)
        # 保留第一个出现的
        df = df.drop_duplicates(subset=['全文'], keep='first')
        logger.info(f"移除重复内容: {before_dedup - len(df)} 条")

    # 5. 重置索引
    df = df.reset_index(drop=True)

    # 6. 重新生成doc_id
    df['doc_id'] = range(1, len(df) + 1)

    # 保存清理后的数据
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)

    # 打印统计信息
    logger.info(f"\n清理完成:")
    logger.info(f"  原始数据: {original_count} 条")
    logger.info(f"  清理后: {len(df)} 条")
    logger.info(f"  保留比例: {len(df) / original_count * 100:.1f}%")

    # 数据质量统计
    print("\n数据质量统计:")
    print(f"  平均文本长度: {df['全文长度'].mean():.0f} 字符")
    print(f"  最短文本: {df['全文长度'].min()} 字符")
    print(f"  最长文本: {df['全文长度'].max()} 字符")
    print(f"  年份范围: {df['年份'].min()}-{df['年份'].max()}")

    if '刊号' in df.columns:
        print(f"  刊号缺失: {df['刊号'].isna().sum()} 个")

    if '分类' in df.columns:
        print(f"\n分类分布:")
        category_counts = df['分类'].value_counts().head(10)
        for cat, count in category_counts.items():
            if cat.strip():  # 忽略空分类
                print(f"    {cat}: {count} ({count / len(df) * 100:.1f}%)")

    return df


def analyze_data_quality(df: pd.DataFrame):
    """分析数据质量"""
    print("\n详细数据质量分析:")

    # 1. 文本长度分布
    print("\n文本长度分布:")
    length_ranges = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 10000), (10000, float('inf'))]
    for start, end in length_ranges:
        if end == float('inf'):
            count = (df['全文长度'] >= start).sum()
            print(f"  >={start}: {count} ({count / len(df) * 100:.1f}%)")
        else:
            count = ((df['全文长度'] >= start) & (df['全文长度'] < end)).sum()
            print(f"  {start}-{end}: {count} ({count / len(df) * 100:.1f}%)")

    # 2. 年份分布
    print("\n年份分布（前10）:")
    year_counts = df['年份'].value_counts().head(10)
    for year, count in year_counts.items():
        print(f"  {year}: {count} ({count / len(df) * 100:.1f}%)")

    # 3. 作者统计
    if '作者名称' in df.columns:
        print(f"\n作者统计:")
        print(f"  唯一作者数: {df['作者名称'].nunique()}")

        # 高产作者
        author_counts = df['作者名称'].value_counts().head(10)
        print("\n  高产作者（前10）:")
        for author, count in author_counts.items():
            if author and str(author).strip():
                print(f"    {author}: {count} 篇")


if __name__ == "__main__":
    # 执行数据清理
    print("开始数据清理...")

    # 使用默认参数清理数据
    clean_df = clean_art_design_data(
        min_text_length=50,  # 至少50字符
        fill_missing_issue=True,  # 填充缺失刊号
        remove_duplicates=True,  # 移除重复
        remove_empty_titles=True  # 移除空标题
    )

    # 分析清理后的数据质量
    analyze_data_quality(clean_df)

    print("\n清理后的数据已保存到: data/processed/processed_data_clean.parquet")