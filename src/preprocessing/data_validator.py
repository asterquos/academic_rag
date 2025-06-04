"""
data_processing/data_validator.py
数据验证和质量检查模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from collections import Counter
import yaml
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器，确保数据质量和完整性"""

    def __init__(self):
        # 加载配置
        import yaml
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        validation_config = config.get('preprocessing', {}).get('validation', {})
        text_config = config.get('preprocessing', {}).get('text', {})

        # 定义验证规则
        self.year_range = tuple(validation_config.get('year_range', [1900, 2030]))
        self.min_text_length = text_config.get('min_text_length', 50)
        self.max_text_length = text_config.get('max_text_length', 100000)

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        综合验证数据框

        Args:
            df: 待验证的数据框

        Returns:
            Dict[str, any]: 验证报告
        """
        report = {
            'total_records': len(df),
            'validation_time': datetime.now().isoformat(),
            'errors': [],
            'warnings': [],
            'statistics': {},
            'data_quality_score': 100.0
        }

        # 识别包含复杂类型的列
        complex_type_cols = []
        for col in df.columns:
            if not df[col].empty:
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample_val, (list, dict, set)):
                    complex_type_cols.append(col)

        # 创建一个只包含简单类型的DataFrame副本用于某些验证
        simple_df = df.drop(columns=complex_type_cols, errors='ignore')

        # 将复杂类型列信息添加到报告中
        if complex_type_cols:
            report['statistics']['complex_type_columns'] = complex_type_cols

        # 1. 基础结构验证
        self._validate_structure(df, report)

        # 2. 数据类型验证
        self._validate_data_types(df, report)

        # 3. 必填字段验证
        self._validate_required_fields(df, report)

        # 4. 数据范围验证
        self._validate_data_ranges(df, report)

        # 5. 数据一致性验证（使用简化的DataFrame）
        self._validate_consistency(simple_df, report)

        # 6. 重复数据检测（使用简化的DataFrame）
        self._detect_duplicates_safe(df, simple_df, report)

        # 7. 异常值检测
        self._detect_outliers(df, report)

        # 8. 生成统计信息
        self._generate_statistics_safe(df, report)

        # 计算数据质量得分
        report['data_quality_score'] = self._calculate_quality_score(report)

        return report

    def _validate_structure(self, df: pd.DataFrame, report: Dict):
        """验证数据框结构"""
        # 检查是否为空
        if df.empty:
            report['errors'].append("数据框为空")
            return

        # 检查列名
        expected_columns = ['年份', '刊号', '分类', '是否入选', '文章名称+副标题', '作者名称', '全文']
        missing_columns = set(expected_columns) - set(df.columns)

        if missing_columns:
            report['errors'].append(f"缺失必需列: {missing_columns}")

        # 检查额外列
        extra_columns = set(df.columns) - set(expected_columns + ['选图情况', '文章特点', '作者介绍'])
        if extra_columns:
            report['warnings'].append(f"发现额外列: {extra_columns}")

    def _validate_data_types(self, df: pd.DataFrame, report: Dict):
        """验证数据类型"""
        type_errors = []

        # 年份应该是数值类型
        if '年份' in df.columns:
            non_numeric = df[pd.to_numeric(df['年份'], errors='coerce').isna() & df['年份'].notna()]
            if len(non_numeric) > 0:
                type_errors.append(f"年份列包含{len(non_numeric)}个非数值数据")

        # 是否入选应该是布尔类型
        if '是否入选' in df.columns:
            # 只检查非列表和非字典的值
            mask = df['是否入选'].apply(lambda x: not isinstance(x, (list, dict)))
            valid_values = [True, False, np.nan, None]
            invalid = df[mask & ~df['是否入选'].isin(valid_values)]
            if len(invalid) > 0:
                type_errors.append(f"是否入选列包含{len(invalid)}个无效值")

        if type_errors:
            report['errors'].extend(type_errors)

    def _validate_required_fields(self, df: pd.DataFrame, report: Dict):
        """验证必填字段"""
        null_counts = {}

        required_fields = ['年份', '刊号', '文章名称+副标题', '作者名称', '全文']

        for field in required_fields:
            if field in df.columns:
                # 对于不同类型的字段使用不同的空值检测方法
                if field in ['作者列表', '句子列表', '概念词', '实体']:
                    # 对于列表或字典类型，检查是否为空列表或空字典
                    null_count = df[field].apply(
                        lambda x: x is None or
                        (isinstance(x, list) and len(x) == 0) or
                        (isinstance(x, dict) and len(x) == 0)
                    ).sum()
                else:
                    # 对于普通字段，使用标准的isna()
                    null_count = df[field].isna().sum()

                if null_count > 0:
                    null_counts[field] = int(null_count)

        if null_counts:
            report['warnings'].append(f"必填字段包含空值: {null_counts}")
            report['statistics']['null_counts'] = null_counts

    def _validate_data_ranges(self, df: pd.DataFrame, report: Dict):
        """验证数据范围"""
        range_errors = []

        # 验证年份范围
        if '年份' in df.columns:
            years = pd.to_numeric(df['年份'], errors='coerce')
            valid_years = years.dropna()

            if len(valid_years) > 0:
                min_year = valid_years.min()
                max_year = valid_years.max()

                if min_year < self.year_range[0] or max_year > self.year_range[1]:
                    range_errors.append(
                        f"年份超出合理范围: {min_year}-{max_year} "
                        f"(期望: {self.year_range[0]}-{self.year_range[1]})"
                    )

        # 验证文本长度
        if '全文' in df.columns:
            text_lengths = df['全文'].str.len()

            too_short = (text_lengths < self.min_text_length).sum()
            too_long = (text_lengths > self.max_text_length).sum()

            if too_short > 0:
                range_errors.append(f"{too_short}篇文章的全文过短（<{self.min_text_length}字符）")
            if too_long > 0:
                range_errors.append(f"{too_long}篇文章的全文过长（>{self.max_text_length}字符）")

        if range_errors:
            report['warnings'].extend(range_errors)

    def _validate_consistency(self, df: pd.DataFrame, report: Dict):
        """验证数据一致性"""
        consistency_issues = []

        # 检查同一刊号是否有不同年份
        if '刊号' in df.columns and '年份' in df.columns:
            # 确保刊号是字符串类型
            df_copy = df.copy()
            df_copy['刊号'] = df_copy['刊号'].astype(str)
            issue_years = df_copy.groupby('刊号')['年份'].nunique()
            inconsistent_issues = issue_years[issue_years > 1]

            if len(inconsistent_issues) > 0:
                consistency_issues.append(
                    f"{len(inconsistent_issues)}个刊号存在多个年份: "
                    f"{list(inconsistent_issues.index[:5])}..."
                )

        # 检查作者名称格式一致性
        if '作者名称' in df.columns:
            # 只处理字符串类型的作者名称
            author_series = df['作者名称'].astype(str)

            # 检查是否混用了不同的分隔符
            separators = ['，', ',', '、', ';', '；']
            sep_usage = {sep: author_series.str.contains(sep, na=False).sum() for sep in separators}

            used_seps = [sep for sep, count in sep_usage.items() if count > 0]
            if len(used_seps) > 1:
                consistency_issues.append(
                    f"作者名称使用了多种分隔符: {used_seps}"
                )

        if consistency_issues:
            report['warnings'].extend(consistency_issues)

    def _detect_duplicates(self, df: pd.DataFrame, report: Dict):
        """检测重复数据"""
        duplicates_info = {}

        # 获取可以用于检测重复的列（排除包含列表、字典的列）
        hashable_columns = []
        for col in df.columns:
            # 检查列是否包含不可哈希的类型
            try:
                # 尝试对该列进行value_counts操作
                df[col].value_counts()
                hashable_columns.append(col)
            except:
                # 如果失败，说明包含不可哈希的类型
                pass

        # 完全重复的行（只使用可哈希的列）
        if hashable_columns:
            full_duplicates = df[hashable_columns].duplicated().sum()
            if full_duplicates > 0:
                duplicates_info['完全重复行'] = full_duplicates

        # 基于标题的重复
        if '文章名称+副标题' in hashable_columns:
            title_duplicates = df.duplicated(subset=['文章名称+副标题']).sum()
            if title_duplicates > 0:
                duplicates_info['标题重复'] = title_duplicates

                # 找出重复的标题
                duplicate_titles = df[df.duplicated(subset=['文章名称+副标题'], keep=False)]
                duplicate_title_groups = duplicate_titles.groupby('文章名称+副标题').size()
                top_duplicates = duplicate_title_groups.nlargest(5)

                duplicates_info['重复标题示例'] = top_duplicates.to_dict()

        # 基于内容的相似度（简化版：完全相同的全文）
        if '全文' in hashable_columns:
            content_duplicates = df.duplicated(subset=['全文']).sum()
            if content_duplicates > 0:
                duplicates_info['内容重复'] = content_duplicates

        if duplicates_info:
            report['warnings'].append(f"检测到重复数据: {duplicates_info}")
            report['statistics']['duplicates'] = duplicates_info

    def _detect_outliers(self, df: pd.DataFrame, report: Dict):
        """检测异常值"""
        outliers = {}

        # 文本长度异常值
        if '全文' in df.columns:
            # 处理可能的NaN值
            lengths = df['全文'].fillna('').astype(str).str.len()

            # 只对非零长度进行统计
            non_zero_lengths = lengths[lengths > 0]
            if len(non_zero_lengths) > 0:
                q1 = non_zero_lengths.quantile(0.25)
                q3 = non_zero_lengths.quantile(0.75)
                iqr = q3 - q1

                lower_bound = max(0, q1 - 1.5 * iqr)  # 确保下界不为负
                upper_bound = q3 + 1.5 * iqr

                outliers_count = ((non_zero_lengths < lower_bound) | (non_zero_lengths > upper_bound)).sum()
                if outliers_count > 0:
                    outliers['文本长度异常'] = {
                        'count': int(outliers_count),
                        'lower_bound': int(lower_bound),
                        'upper_bound': int(upper_bound)
                    }

        # 年份分布异常
        if '年份' in df.columns:
            # 确保年份是可处理的类型
            year_series = pd.to_numeric(df['年份'], errors='coerce').dropna()
            if len(year_series) > 0:
                year_counts = year_series.value_counts()
                if len(year_counts) > 1:  # 至少有两个不同年份
                    mean_count = year_counts.mean()
                    std_count = year_counts.std()

                    if std_count > 0:  # 避免除零错误
                        # 找出文章数量异常多或异常少的年份
                        abnormal_years = year_counts[
                            (year_counts > mean_count + 2 * std_count) |
                            (year_counts < mean_count - 2 * std_count)
                        ]

                        if len(abnormal_years) > 0:
                            outliers['年份分布异常'] = {
                                str(int(year)): int(count)
                                for year, count in abnormal_years.items()
                            }

        if outliers:
            report['statistics']['outliers'] = outliers

    def _detect_duplicates_safe(self, df: pd.DataFrame, simple_df: pd.DataFrame, report: Dict):
        """安全地检测重复数据"""
        duplicates_info = {}

        # 使用简化的DataFrame检测完全重复
        if not simple_df.empty:
            full_duplicates = simple_df.duplicated().sum()
            if full_duplicates > 0:
                duplicates_info['完全重复行（不含复杂列）'] = int(full_duplicates)

        # 基于标题的重复
        if '文章名称+副标题' in df.columns:
            title_duplicates = df.duplicated(subset=['文章名称+副标题']).sum()
            if title_duplicates > 0:
                duplicates_info['标题重复'] = int(title_duplicates)

                # 找出重复的标题（只显示前5个）
                duplicate_titles = df[df.duplicated(subset=['文章名称+副标题'], keep=False)]
                if len(duplicate_titles) > 0:
                    # 安全地获取标题统计
                    title_counts = {}
                    for title in duplicate_titles['文章名称+副标题']:
                        title_str = str(title)
                        title_counts[title_str] = title_counts.get(title_str, 0) + 1

                    # 排序并取前5个
                    top_titles = sorted(title_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    duplicates_info['重复标题示例'] = dict(top_titles)

        # 基于内容的重复（如果全文是简单类型）
        if '全文' in simple_df.columns:
            content_duplicates = df.duplicated(subset=['全文']).sum()
            if content_duplicates > 0:
                duplicates_info['内容重复'] = int(content_duplicates)

        if duplicates_info:
            report['warnings'].append(f"检测到重复数据")
            report['statistics']['duplicates'] = duplicates_info

    def _generate_statistics_safe(self, df: pd.DataFrame, report: Dict):
        """安全地生成统计信息"""
        stats = report['statistics']

        # 基础统计
        stats['总记录数'] = len(df)

        if '年份' in df.columns:
            # 安全地处理年份
            try:
                years = pd.to_numeric(df['年份'], errors='coerce')
                valid_years = years.dropna()
                if len(valid_years) > 0:
                    stats['年份范围'] = f"{int(valid_years.min())}-{int(valid_years.max())}"
                    # 只取前10个年份的分布
                    year_counts = valid_years.value_counts().head(10)
                    stats['年份分布'] = {str(int(k)): int(v) for k, v in year_counts.items()}
            except Exception as e:
                logger.warning(f"年份统计失败: {e}")

        if '分类' in df.columns:
            # 安全地处理分类
            try:
                # 确保分类是字符串
                categories = df['分类'].fillna('未分类').astype(str)
                stats['分类统计'] = categories.value_counts().to_dict()
            except Exception as e:
                logger.warning(f"分类统计失败: {e}")

        if '作者名称' in df.columns:
            # 简单统计不同作者数
            try:
                stats['作者数量'] = df['作者名称'].nunique()
            except:
                stats['作者数量'] = '统计失败'

        if '全文' in df.columns:
            try:
                # 安全地计算文本长度
                lengths = df['全文'].fillna('').astype(str).str.len()
                stats['文本长度统计'] = {
                    '平均长度': int(lengths.mean()) if len(lengths) > 0 else 0,
                    '最短': int(lengths.min()) if len(lengths) > 0 else 0,
                    '最长': int(lengths.max()) if len(lengths) > 0 else 0,
                    '中位数': int(lengths.median()) if len(lengths) > 0 else 0
                }
            except Exception as e:
                logger.warning(f"文本长度统计失败: {e}")

        if '是否入选' in df.columns:
            try:
                # 安全地处理布尔值
                selected_series = df['是否入选'].fillna(False)
                if selected_series.dtype == bool:
                    selected_count = selected_series.sum()
                else:
                    # 尝试转换为布尔值
                    selected_count = (selected_series == True).sum()

                stats['入选统计'] = {
                    '入选数': int(selected_count),
                    '未入选数': len(df) - int(selected_count),
                    '入选率': f"{selected_count / len(df) * 100:.2f}%" if len(df) > 0 else "0%"
                }
            except Exception as e:
                logger.warning(f"入选统计失败: {e}")
            
    def _calculate_quality_score(self, report: Dict) -> float:
        """
        计算数据质量得分
        
        Args:
            report: 验证报告
            
        Returns:
            float: 质量得分（0-100）
        """
        score = 100.0
        
        # 错误扣分（每个错误-10分）
        score -= len(report['errors']) * 10
        
        # 警告扣分（每个警告-5分）
        score -= len(report['warnings']) * 5
        
        # 空值扣分
        if 'null_counts' in report['statistics']:
            null_ratio = sum(report['statistics']['null_counts'].values()) / (
                report['total_records'] * len(report['statistics']['null_counts'])
            )
            score -= null_ratio * 20
            
        # 重复数据扣分
        if 'duplicates' in report['statistics']:
            dup_count = report['statistics']['duplicates'].get('完全重复行', 0)
            dup_ratio = dup_count / report['total_records']
            score -= dup_ratio * 30
            
        return max(0.0, score)
        
    def generate_report_summary(self, report: Dict) -> str:
        """
        生成验证报告摘要
        
        Args:
            report: 验证报告
            
        Returns:
            str: 格式化的报告摘要
        """
        summary = f"""
数据验证报告
================
验证时间: {report['validation_time']}
总记录数: {report['total_records']}
数据质量得分: {report['data_quality_score']:.2f}/100

错误 ({len(report['errors'])}):
"""
        for error in report['errors']:
            summary += f"  - {error}\n"
            
        summary += f"\n警告 ({len(report['warnings'])}):\n"
        for warning in report['warnings']:
            summary += f"  - {warning}\n"
            
        summary += "\n统计信息:\n"
        for key, value in report['statistics'].items():
            if isinstance(value, dict):
                summary += f"  {key}:\n"
                for k, v in value.items():
                    summary += f"    - {k}: {v}\n"
            else:
                summary += f"  {key}: {value}\n"
                
        return summary


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    sample_data = pd.DataFrame({
        '年份': [2020, 2020, 2021, 2021, None],
        '刊号': ['2020-01', '2020-01', '2021-01', '2021-02', '2021-02'],
        '分类': ['设计理论', '设计实践', '设计理论', '设计批评', '设计实践'],
        '是否入选': [True, True, False, True, None],
        '文章名称+副标题': ['包豪斯的现代性', '现代主义设计', '后现代设计思潮', '设计的未来', '包豪斯的现代性'],
        '作者名称': ['张三', '李四，王五', '赵六', None, '张三'],
        '全文': ['这是一篇关于包豪斯的文章...' * 10] * 4 + ['短文本']
    })
    
    # 创建验证器
    validator = DataValidator()
    
    # 执行验证
    report = validator.validate_dataframe(sample_data)
    
    # 打印报告摘要
    print(validator.generate_report_summary(report))