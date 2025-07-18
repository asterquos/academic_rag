"""
src/preprocessing/pipeline.py
数据预处理管道主程序
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml

# 导入各个处理模块
from .excel_parser import ExcelParser
from .text_processor import TextProcessor
from .data_validator import DataValidator

logger = logging.getLogger(__name__)


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class PreprocessingPipeline:
    """数据预处理管道，整合所有预处理步骤"""

    def __init__(self):
        """初始化预处理管道"""
        # 加载配置
        self.config = load_config()
        self.preprocessing_config = self.config.get('preprocessing', {})

        # 设置日志
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        # 初始化各个处理器
        self.excel_parser = ExcelParser()
        self.text_processor = TextProcessor()
        self.validator = DataValidator()

        # 创建输出目录
        self.output_dir = Path(self.preprocessing_config.get('pipeline', {}).get('output_dir', 'data/processed'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("预处理管道初始化完成")

    def run(self, input_file: str) -> Dict[str, any]:
        """
        运行完整的预处理管道

        Args:
            input_file: 输入的Excel文件路径

        Returns:
            Dict[str, any]: 处理结果和统计信息
        """
        logger.info(f"开始处理文件: {input_file}")
        start_time = datetime.now()

        results = {
            'input_file': input_file,
            'start_time': start_time.isoformat(),
            'status': 'processing',
            'steps': {}
        }

        try:
            # Step 1: Excel解析和基础处理
            logger.info("Step 1: Excel解析和基础处理")
            df = self._parse_excel(input_file)
            results['steps']['excel_parsing'] = {
                'status': 'completed',
                'records': len(df),
                'columns': list(df.columns)
            }

            # Step 2: 文本处理
            logger.info("Step 2: 文本处理")
            df = self._process_text(df)
            results['steps']['text_processing'] = {
                'status': 'completed',
                'new_columns': [col for col in df.columns if col.endswith('_清洗') or col in ['句子列表', '概念词', '实体']]
            }

            # Step 3: 数据验证
            validation_report = None
            if self.preprocessing_config.get('validation', {}).get('enable_validation', True):
                logger.info("Step 3: 数据验证")
                validation_report = self._validate_data(df)
                results['steps']['validation'] = {
                    'status': 'completed',
                    'quality_score': validation_report['data_quality_score'],
                    'errors': len(validation_report['errors']),
                    'warnings': len(validation_report['warnings'])
                }

            # Step 4: 创建文本块
            logger.info("Step 4: 创建文本块")
            chunks_df = self._create_chunks(df)
            results['steps']['chunking'] = {
                'status': 'completed',
                'total_chunks': len(chunks_df),
                'avg_chunks_per_doc': len(chunks_df) / len(df) if len(df) > 0 else 0
            }

            # Step 5: 准备向量化数据
            logger.info("Step 5: 准备向量化数据")
            vector_data = self._prepare_vector_data(df, chunks_df)
            results['steps']['vector_preparation'] = {
                'status': 'completed',
                'documents': len(vector_data['documents']),
                'chunks': len(vector_data['chunks'])
            }

            # Step 6: 保存处理结果
            logger.info("Step 6: 保存处理结果")
            output_files = self._save_results(df, chunks_df, vector_data, validation_report)
            results['steps']['saving'] = {
                'status': 'completed',
                'output_files': output_files
            }

            # 完成处理
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['duration'] = str(end_time - start_time)
            results['status'] = 'completed'

            logger.info(f"处理完成，总耗时: {results['duration']}")

        except Exception as e:
            logger.error(f"处理失败: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)

        return results

    def _parse_excel(self, input_file: str) -> pd.DataFrame:
        """Excel解析步骤"""
        df = self.excel_parser.process(
            input_file,
            filter_selected=self.preprocessing_config.get('pipeline', {}).get('filter_selected', False)
        )

        if self.preprocessing_config.get('pipeline', {}).get('save_intermediate', True):
            output_file = self.output_dir / 'step1_parsed.parquet'
            df.to_parquet(output_file)
            logger.info(f"保存中间结果: {output_file}")

        return df

    def _process_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """文本处理步骤"""
        df = self.text_processor.process_dataframe(df)

        if self.preprocessing_config.get('pipeline', {}).get('save_intermediate', True):
            output_file = self.output_dir / 'step2_text_processed.parquet'
            df.to_parquet(output_file)
            logger.info(f"保存中间结果: {output_file}")

        return df

    def _validate_data(self, df: pd.DataFrame) -> Dict:
        """数据验证步骤"""
        report = self.validator.validate_dataframe(df)

        if self.preprocessing_config.get('pipeline', {}).get('save_intermediate', True):
            output_file = self.output_dir / 'step3_validation_report.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"保存验证报告: {output_file}")

        # 打印验证摘要
        summary = self.validator.generate_report_summary(report)
        logger.info(f"\n{summary}")

        return report

    def _create_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建文本块"""
        chunks_data = []

        text_config = self.preprocessing_config.get('text', {})
        chunk_size = text_config.get('chunk_size', 500)
        chunk_overlap = text_config.get('chunk_overlap', 50)

        for idx, row in df.iterrows():
            # 检查是否有清洗后的全文
            text_field = '全文_清洗' if '全文_清洗' in row else '全文'

            if text_field not in row or pd.isna(row[text_field]):
                continue

            # 创建文本块
            chunks = self.text_processor.create_text_chunks(
                row[text_field],
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )

            # 为每个块添加元数据
            for chunk in chunks:
                chunk_record = {
                    'doc_id': row.get('doc_id', idx),
                    'chunk_id': f"{row.get('doc_id', idx)}_{chunk['chunk_id']}",
                    'chunk_text': chunk['text'],
                    'chunk_position': chunk['chunk_id'],
                    'year': row.get('年份'),
                    'title': row.get('主标题', row.get('文章名称+副标题', '')),
                    'author': row.get('作者名称'),
                    'category': row.get('分类')
                }
                chunks_data.append(chunk_record)

        chunks_df = pd.DataFrame(chunks_data)

        if self.preprocessing_config.get('pipeline', {}).get('save_intermediate', True):
            output_file = self.output_dir / 'step4_chunks.parquet'
            chunks_df.to_parquet(output_file)
            logger.info(f"保存文本块: {output_file}")

        return chunks_df

    def _prepare_vector_data(self, df: pd.DataFrame, chunks_df: pd.DataFrame) -> Dict:
        """准备向量化数据"""
        vector_data = {
            'documents': [],
            'chunks': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_documents': len(df),
                'total_chunks': len(chunks_df)
            }
        }

        # 准备文档级数据
        for idx, row in df.iterrows():
            # 获取文本内容
            text_content = row.get('全文_清洗', row.get('全文', ''))

            doc_data = {
                'doc_id': str(row.get('doc_id', idx)),
                'content': text_content,
                'metadata': {
                    'year': int(row['年份']) if pd.notna(row.get('年份')) else None,
                    'title': row.get('主标题', row.get('文章名称+副标题', '')),
                    'subtitle': row.get('副标题', ''),
                    'authors': row.get('作者列表', []),
                    'category': row.get('分类', ''),
                    'journal_issue': row.get('刊号', ''),
                    'is_selected': row.get('是否入选', False),
                    'concepts': row.get('概念词', []),
                    'entities': row.get('实体', [])
                }
            }
            vector_data['documents'].append(doc_data)

        # 准备块级数据
        for idx, row in chunks_df.iterrows():
            chunk_data = {
                'chunk_id': row['chunk_id'],
                'doc_id': str(row['doc_id']),
                'content': row['chunk_text'],
                'metadata': {
                    'chunk_position': row['chunk_position'],
                    'year': row.get('year'),
                    'title': row.get('title'),
                    'author': row.get('author'),
                    'category': row.get('category')
                }
            }
            vector_data['chunks'].append(chunk_data)

        return vector_data

    def _save_results(self, df: pd.DataFrame, chunks_df: pd.DataFrame,
                     vector_data: Dict, validation_report: Optional[Dict] = None) -> List[str]:
        """保存所有处理结果"""
        output_files = []

        # 1. 保存完整处理后的数据
        output_file = self.output_dir / 'processed_data.parquet'
        df.to_parquet(output_file)
        output_files.append(str(output_file))
        logger.info(f"保存处理后数据: {output_file}")

        # 2. 保存文本块数据
        output_file = self.output_dir / 'chunks_data.parquet'
        chunks_df.to_parquet(output_file)
        output_files.append(str(output_file))
        logger.info(f"保存文本块数据: {output_file}")

        # 3. 保存向量化数据（JSON格式）
        output_file = self.output_dir / 'vector_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, ensure_ascii=False, indent=2)
        output_files.append(str(output_file))
        logger.info(f"保存向量化数据: {output_file}")

        # 4. 保存验证报告
        if validation_report:
            output_file = self.output_dir / 'validation_report.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, ensure_ascii=False, indent=2)
            output_files.append(str(output_file))

        # 5. 生成数据摘要
        summary = self._generate_summary(df, chunks_df, validation_report)
        output_file = self.output_dir / 'data_summary.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        output_files.append(str(output_file))
        logger.info(f"保存数据摘要: {output_file}")

        return output_files

    def _generate_summary(self, df: pd.DataFrame, chunks_df: pd.DataFrame,
                         validation_report: Optional[Dict] = None) -> Dict:
        """生成数据摘要"""
        summary = {
            'processing_time': datetime.now().isoformat(),
            'statistics': {
                'total_documents': len(df),
                'total_chunks': len(chunks_df),
                'year_range': f"{df['年份'].min()}-{df['年份'].max()}" if '年份' in df.columns else None,
                'categories': df['分类'].value_counts().to_dict() if '分类' in df.columns else {},
                'authors_count': df['作者名称'].nunique() if '作者名称' in df.columns else 0,
                'selected_ratio': f"{(df['是否入选'].sum() / len(df) * 100):.2f}%" if '是否入选' in df.columns else None,
                'avg_text_length': int(df['全文长度'].mean()) if '全文长度' in df.columns else 0,
                'avg_chunks_per_doc': len(chunks_df) / len(df) if len(df) > 0 else 0
            }
        }

        if validation_report:
            summary['data_quality'] = {
                'score': validation_report['data_quality_score'],
                'errors': len(validation_report['errors']),
                'warnings': len(validation_report['warnings'])
            }

        return summary


def main():
    """主程序入口"""
    import argparse

    parser = argparse.ArgumentParser(description='艺术设计文献数据预处理管道')
    parser.add_argument('--input', help='输入的Excel文件路径（默认使用config.yaml中的路径）')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--filter-selected', action='store_true', help='只处理入选的文章')
    parser.add_argument('--no-validation', action='store_true', help='跳过数据验证')
    parser.add_argument('--chunk-size', type=int, help='文本块大小')
    parser.add_argument('--chunk-overlap', type=int, help='文本块重叠大小')

    args = parser.parse_args()

    # 创建管道
    pipeline = PreprocessingPipeline()

    # 从命令行参数覆盖配置
    if args.output_dir:
        pipeline.output_dir = Path(args.output_dir)
        pipeline.output_dir.mkdir(parents=True, exist_ok=True)

    if args.filter_selected:
        pipeline.preprocessing_config.setdefault('pipeline', {})['filter_selected'] = True

    if args.no_validation:
        pipeline.preprocessing_config.setdefault('validation', {})['enable_validation'] = False

    if args.chunk_size:
        pipeline.preprocessing_config.setdefault('text', {})['chunk_size'] = args.chunk_size

    if args.chunk_overlap:
        pipeline.preprocessing_config.setdefault('text', {})['chunk_overlap'] = args.chunk_overlap

    # 确定输入文件
    input_file = args.input or pipeline.config.get('data', {}).get('raw_path')
    if not input_file:
        parser.error("请指定输入文件或在config.yaml中配置data.raw_path")

    # 运行管道
    results = pipeline.run(input_file)

    # 打印结果
    print("\n处理结果:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 示例用法 - 直接使用config.yaml配置
    pipeline = PreprocessingPipeline()

    # 从配置文件获取输入路径
    config = load_config()
    input_file = config.get('data', {}).get('raw_path', 'data/raw/applied_arts.xlsx')

    results = pipeline.run(input_file)

    print("\n处理完成！")
    print(f"处理状态: {results['status']}")
    
    # 检查是否有duration字段
    if 'duration' in results:
        print(f"处理时间: {results['duration']}")
    
    if results['status'] == 'completed':
        print("\n各步骤结果:")
        for step, info in results['steps'].items():
            print(f"  {step}: {info['status']}")

        if 'saving' in results['steps'] and 'output_files' in results['steps']['saving']:
            print("\n输出文件:")
            for file in results['steps']['saving']['output_files']:
                print(f"  - {file}")
    elif results['status'] == 'failed' and 'error' in results:
        print(f"错误信息: {results['error']}")