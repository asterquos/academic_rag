"""
data_processing/excel_parser.py
Excel数据解析和预处理模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelParser:
    """Excel文件解析器，处理艺术设计文献数据"""

    def __init__(self, encoding: str = None):
        # 加载配置
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        excel_config = config.get('preprocessing', {}).get('excel', {})

        self.encoding = encoding or excel_config.get('encoding', 'utf-8')
        self.required_columns = excel_config.get('required_columns', [
            '年份', '刊号', '分类', '是否入选',
            '文章名称+副标题', '作者名称', '全文'
        ])
        self.optional_columns = excel_config.get('optional_columns', ['选图情况', '文章特点', '作者介绍'])
        self.supported_encodings = excel_config.get('supported_encodings', ['utf-8', 'gbk', 'gb2312', 'cp1252'])

    def load_excel(self, file_path: str) -> pd.DataFrame:
        """
        加载Excel文件

        Args:
            file_path: Excel文件路径

        Returns:
            pd.DataFrame: 原始数据框
        """
        try:
            # 使用配置的编码列表
            for enc in self.supported_encodings:
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    logger.info(f"成功加载Excel文件: {file_path}")
                    logger.info(f"数据形状: {df.shape}")
                    return df
                except Exception as e:
                    continue

            raise ValueError(f"无法读取Excel文件: {file_path}")

        except Exception as e:
            logger.error(f"加载Excel文件失败: {e}")
            raise

    def validate_columns(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证必需列是否存在

        Args:
            df: 数据框

        Returns:
            (是否有效, 缺失列列表)
        """
        missing_columns = []
        for col in self.required_columns:
            if col not in df.columns:
                missing_columns.append(col)

        if missing_columns:
            logger.warning(f"缺失必需列: {missing_columns}")

        return len(missing_columns) == 0, missing_columns

    def fill_year_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        填充年份值 - 每本杂志年份只出现一次，需要填充到所有相关行

        Args:
            df: 原始数据框

        Returns:
            pd.DataFrame: 年份填充后的数据框
        """
        df = df.copy()

        # 方法1: 基于刊号分组填充
        if '刊号' in df.columns and '年份' in df.columns:
            # 先尝试前向填充
            df['年份'] = df.groupby('刊号')['年份'].ffill()
            # 再尝试后向填充（处理年份在组末尾的情况）
            df['年份'] = df.groupby('刊号')['年份'].bfill()

        # 方法2: 如果还有空值，使用全局前向填充
        if df['年份'].isna().any():
            df['年份'] = df['年份'].ffill()

        # 转换为整数类型
        df['年份'] = pd.to_numeric(df['年份'], errors='coerce').astype('Int64')

        logger.info(f"年份填充完成，空值数量: {df['年份'].isna().sum()}")
        return df

    def clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理文本字段

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 清理后的数据框
        """
        df = df.copy()

        text_fields = ['文章名称+副标题', '作者名称', '全文', '分类', '文章特点', '作者介绍']

        for field in text_fields:
            if field in df.columns:
                # 去除前后空白
                df[field] = df[field].astype(str).str.strip()
                # 替换多个空格为单个空格
                df[field] = df[field].str.replace(r'\s+', ' ', regex=True)
                # 将'nan'字符串替换为空字符串
                df[field] = df[field].replace('nan', '')

        return df

    def process_boolean_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理布尔字段

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        df = df.copy()

        if '是否入选' in df.columns:
            # 处理各种可能的布尔值表示
            true_values = ['是', '1', 'True', 'true', 'Y', 'y', '入选']
            false_values = ['否', '0', 'False', 'false', 'N', 'n', '未入选']

            df['是否入选'] = df['是否入选'].astype(str).str.strip()
            df.loc[df['是否入选'].isin(true_values), '是否入选'] = True
            df.loc[df['是否入选'].isin(false_values), '是否入选'] = False

            # 将其他值设为None
            df.loc[~df['是否入选'].isin([True, False]), '是否入选'] = None

        return df

    def split_title_subtitle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分离主标题和副标题

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加了主标题和副标题列的数据框
        """
        df = df.copy()

        if '文章名称+副标题' in df.columns:
            # 尝试多种分隔符
            separators = ['：', ':', '——', '--', '－', '-']

            df['主标题'] = df['文章名称+副标题']
            df['副标题'] = ''

            for sep in separators:
                mask = df['文章名称+副标题'].str.contains(sep, regex=False, na=False)
                if mask.any():
                    split_result = df.loc[mask, '文章名称+副标题'].str.split(sep, n=1, expand=True)
                    df.loc[mask, '主标题'] = split_result[0].str.strip()
                    if len(split_result.columns) > 1:
                        df.loc[mask, '副标题'] = split_result[1].str.strip()

        return df

    def extract_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取和标准化作者信息

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加了作者列表的数据框
        """
        df = df.copy()

        if '作者名称' in df.columns:
            # 处理多作者情况
            separators = ['，', ',', '、', ';', '；', ' and ', ' & ']

            def split_authors(author_str):
                if pd.isna(author_str) or author_str == '':
                    return []

                authors = [author_str]
                for sep in separators:
                    new_authors = []
                    for author in authors:
                        new_authors.extend(author.split(sep))
                    authors = new_authors

                # 清理和去重
                authors = [a.strip() for a in authors if a.strip()]
                return list(dict.fromkeys(authors))  # 保持顺序的去重

            df['作者列表'] = df['作者名称'].apply(split_authors)
            df['作者数量'] = df['作者列表'].apply(len)

        return df

    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加元数据字段

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加了元数据的数据框
        """
        df = df.copy()

        # 添加文档ID
        df['doc_id'] = range(1, len(df) + 1)

        # 添加文本长度
        if '全文' in df.columns:
            df['全文长度'] = df['全文'].str.len()

        # 添加处理时间戳
        df['处理时间'] = pd.Timestamp.now()

        return df

    def process(self, file_path: str, filter_selected: bool = False) -> pd.DataFrame:
        """
        完整的数据处理流程

        Args:
            file_path: Excel文件路径
            filter_selected: 是否只保留入选的文章

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        logger.info("开始数据处理流程...")

        # 1. 加载数据
        df = self.load_excel(file_path)

        # 2. 验证列
        valid, missing = self.validate_columns(df)
        if not valid:
            raise ValueError(f"数据验证失败，缺失必需列: {missing}")

        # 3. 填充年份
        df = self.fill_year_values(df)
        
        # 4. 清理文本
        df = self.clean_text_fields(df)
        
        # 5. 处理布尔字段
        df = self.process_boolean_fields(df)
        
        # 6. 分离标题
        df = self.split_title_subtitle(df)
        
        # 7. 提取作者
        df = self.extract_authors(df)
        
        # 8. 添加元数据
        df = self.add_metadata(df)
        
        # 9. 筛选入选文章（如果需要）
        if filter_selected and '是否入选' in df.columns:
            original_count = len(df)
            df = df[df['是否入选'] == True]
            logger.info(f"筛选入选文章: {original_count} -> {len(df)}")
            
        # 10. 重置索引
        df = df.reset_index(drop=True)
        
        logger.info(f"数据处理完成，最终数据形状: {df.shape}")
        return df
        
    def save_processed_data(self, df: pd.DataFrame, output_path: str, format: str = 'parquet'):
        """
        保存处理后的数据
        
        Args:
            df: 处理后的数据框
            output_path: 输出路径
            format: 输出格式 ('parquet', 'csv', 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif format == 'excel':
            df.to_excel(output_path, index=False, engine='openpyxl')
        else:
            raise ValueError(f"不支持的格式: {format}")
            
        logger.info(f"数据已保存到: {output_path}")

    def fill_issue_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        填充刊号值 - 类似年份填充，每个刊号可能只出现一次

        Args:
            df: 原始数据框

        Returns:
            pd.DataFrame: 刊号填充后的数据框
        """
        df = df.copy()

        if '刊号' in df.columns:
            # 方法1: 前向填充（假设同一期的文章是连续的）
            df['刊号'] = df['刊号'].ffill()

            # 方法2: 如果还有空值，可能是开头的空值，尝试后向填充
            if df['刊号'].isna().any():
                df['刊号'] = df['刊号'].bfill()

            logger.info(f"刊号填充完成，空值数量: {df['刊号'].isna().sum()}")

        return df

# 使用示例
if __name__ == "__main__":
    parser = ExcelParser()
    
    # 处理数据
    df = parser.process("data/art_design_literature.xlsx", filter_selected=True)
    
    # 显示基本信息
    print(f"处理后数据形状: {df.shape}")
    print(f"年份范围: {df['年份'].min()} - {df['年份'].max()}")
    print(f"作者总数: {df['作者列表'].explode().nunique()}")
    print(f"分类统计:\n{df['分类'].value_counts()}")
    
    # 保存处理后的数据
    parser.save_processed_data(df, "data/processed/art_design_processed.parquet")