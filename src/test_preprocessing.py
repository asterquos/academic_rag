"""
test_preprocessing.py
数据预处理模块专项测试
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.excel_parser import ExcelParser
from src.preprocessing.text_processor import TextProcessor
from src.preprocessing.data_validator import DataValidator
from src.preprocessing.pipeline import PreprocessingPipeline

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PreprocessingTester:
    """数据预处理测试器"""
    
    def __init__(self, config_path="config.yaml"):
        """初始化测试器"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
    def test_excel_parser(self):
        """测试Excel解析器"""
        print("\n" + "="*60)
        print("测试 Excel 解析器")
        print("="*60)
        
        parser = ExcelParser()
        results = {'status': 'testing', 'details': {}}
        
        try:
            # 1. 测试加载
            print("\n1. 测试Excel加载...")
            start_time = time.time()
            df = parser.load_excel(self.config['data']['raw_path'])
            load_time = time.time() - start_time
            
            results['details']['load_time'] = f"{load_time:.2f}秒"
            results['details']['shape'] = df.shape
            results['details']['columns'] = list(df.columns)
            
            print(f"✓ 加载成功")
            print(f"  - 耗时: {load_time:.2f}秒")
            print(f"  - 数据量: {df.shape[0]}行 × {df.shape[1]}列")
            
            # 2. 测试列验证
            print("\n2. 测试列验证...")
            valid, missing = parser.validate_columns(df)
            results['details']['columns_valid'] = valid
            results['details']['missing_columns'] = missing
            
            if valid:
                print("✓ 所有必需列都存在")
            else:
                print(f"✗ 缺失列: {missing}")
                
            # 3. 测试年份填充
            print("\n3. 测试年份填充...")
            year_nulls_before = df['年份'].isna().sum()
            df_filled = parser.fill_year_values(df)
            year_nulls_after = df_filled['年份'].isna().sum()
            
            results['details']['year_fill'] = {
                'before': int(year_nulls_before),
                'after': int(year_nulls_after),
                'filled': int(year_nulls_before - year_nulls_after)
            }
            
            print(f"✓ 年份填充完成")
            print(f"  - 填充前空值: {year_nulls_before}")
            print(f"  - 填充后空值: {year_nulls_after}")
            print(f"  - 填充数量: {year_nulls_before - year_nulls_after}")
            
            # 4. 测试完整处理流程
            print("\n4. 测试完整处理流程...")
            start_time = time.time()
            processed_df = parser.process(self.config['data']['raw_path'])
            process_time = time.time() - start_time
            
            results['details']['process_time'] = f"{process_time:.2f}秒"
            results['details']['processed_shape'] = processed_df.shape
            
            print(f"✓ 处理完成")
            print(f"  - 耗时: {process_time:.2f}秒")
            print(f"  - 输出数据: {processed_df.shape[0]}行")
            
            # 5. 数据质量检查
            print("\n5. 数据质量检查...")
            quality_checks = {
                '标题为空': processed_df['文章名称+副标题'].isna().sum(),
                '作者为空': processed_df['作者名称'].isna().sum(),
                '全文为空': processed_df['全文'].isna().sum(),
                '年份为空': processed_df['年份'].isna().sum()
            }
            
            results['details']['quality_checks'] = quality_checks
            
            for check, count in quality_checks.items():
                status = "✓" if count == 0 else "⚠️"
                print(f"  {status} {check}: {count}")
                
            results['status'] = 'passed'
            results['processed_df'] = processed_df
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"\n✗ 测试失败: {e}")
            
        self.results['tests']['excel_parser'] = results
        return results.get('processed_df')
        
    def test_text_processor(self, df=None):
        """测试文本处理器"""
        print("\n" + "="*60)
        print("测试文本处理器")
        print("="*60)
        
        if df is None:
            print("⚠️  没有输入数据，跳过测试")
            return None
            
        processor = TextProcessor()
        results = {'status': 'testing', 'details': {}}
        
        try:
            # 1. 测试文本清洗
            print("\n1. 测试文本清洗...")
            sample_text = df['全文'].iloc[0] if len(df) > 0 else ""
            cleaned_text = processor.clean_text(sample_text)
            
            results['details']['text_cleaning'] = {
                'original_length': len(sample_text),
                'cleaned_length': len(cleaned_text),
                'reduction': f"{(1 - len(cleaned_text)/len(sample_text))*100:.1f}%"
            }
            
            print(f"✓ 文本清洗测试完成")
            print(f"  - 原始长度: {len(sample_text)}")
            print(f"  - 清洗后长度: {len(cleaned_text)}")
            
            # 2. 测试分词
            print("\n2. 测试分词功能...")
            words = processor.segment_text(cleaned_text[:200])
            results['details']['segmentation'] = {
                'sample_words': words[:10],
                'word_count': len(words)
            }
            
            print(f"✓ 分词测试完成")
            print(f"  - 词数: {len(words)}")
            print(f"  - 示例: {' | '.join(words[:10])}")
            
            # 3. 测试概念提取
            print("\n3. 测试概念提取...")
            concepts = processor.extract_concepts(cleaned_text)
            results['details']['concepts'] = {
                'count': len(concepts),
                'top_concepts': list(concepts.items())[:5]
            }
            
            print(f"✓ 概念提取完成")
            print(f"  - 概念数: {len(concepts)}")
            if concepts:
                print(f"  - Top 5: {list(concepts.keys())[:5]}")
                
            # 4. 测试文本分块
            print("\n4. 测试文本分块...")
            chunks = processor.create_text_chunks(cleaned_text, chunk_size=500, overlap=50)
            results['details']['chunking'] = {
                'chunk_count': len(chunks),
                'avg_chunk_size': np.mean([len(c['text']) for c in chunks]) if chunks else 0
            }
            
            print(f"✓ 文本分块完成")
            print(f"  - 块数: {len(chunks)}")
            print(f"  - 平均块大小: {results['details']['chunking']['avg_chunk_size']:.0f}字符")
            
            # 5. 批量处理测试
            print("\n5. 测试批量处理...")
            sample_df = df.head(10).copy()
            start_time = time.time()
            processed_df = processor.process_dataframe(sample_df)
            process_time = time.time() - start_time
            
            results['details']['batch_processing'] = {
                'sample_size': len(sample_df),
                'process_time': f"{process_time:.2f}秒",
                'new_columns': [col for col in processed_df.columns if col not in sample_df.columns]
            }
            
            print(f"✓ 批量处理完成")
            print(f"  - 处理{len(sample_df)}条数据耗时: {process_time:.2f}秒")
            print(f"  - 新增列: {results['details']['batch_processing']['new_columns']}")
            
            results['status'] = 'passed'
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"\n✗ 测试失败: {e}")
            
        self.results['tests']['text_processor'] = results
        return results
        
    def test_data_validator(self, df=None):
        """测试数据验证器"""
        print("\n" + "="*60)
        print("测试数据验证器")
        print("="*60)
        
        if df is None:
            print("⚠️  没有输入数据，跳过测试")
            return None
            
        validator = DataValidator()
        results = {'status': 'testing', 'details': {}}
        
        try:
            print("\n执行数据验证...")
            start_time = time.time()
            report = validator.validate_dataframe(df)
            validate_time = time.time() - start_time
            
            results['details']['validation_time'] = f"{validate_time:.2f}秒"
            results['details']['quality_score'] = report['data_quality_score']
            results['details']['errors'] = len(report['errors'])
            results['details']['warnings'] = len(report['warnings'])
            
            print(f"\n✓ 验证完成")
            print(f"  - 耗时: {validate_time:.2f}秒")
            print(f"  - 数据质量得分: {report['data_quality_score']:.1f}/100")
            print(f"  - 错误数: {len(report['errors'])}")
            print(f"  - 警告数: {len(report['warnings'])}")
            
            # 显示具体问题
            if report['errors']:
                print(f"\n错误:")
                for error in report['errors'][:3]:
                    print(f"  - {error}")
                    
            if report['warnings']:
                print(f"\n警告:")
                for warning in report['warnings'][:3]:
                    print(f"  - {warning}")
                    
            # 显示统计信息
            if 'statistics' in report:
                print(f"\n统计信息:")
                stats = report['statistics']
                
                if 'null_counts' in stats:
                    print(f"  空值统计:")
                    for field, count in stats['null_counts'].items():
                        print(f"    - {field}: {count}")
                        
                if 'duplicates' in stats:
                    print(f"  重复数据:")
                    for key, value in stats['duplicates'].items():
                        print(f"    - {key}: {value}")
                        
            results['status'] = 'passed'
            results['validation_report'] = report
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"\n✗ 测试失败: {e}")
            
        self.results['tests']['data_validator'] = results
        return results
        
    def test_pipeline(self):
        """测试完整预处理管道"""
        print("\n" + "="*60)
        print("测试完整预处理管道")
        print("="*60)
        
        results = {'status': 'testing', 'details': {}}
        
        try:
            pipeline = PreprocessingPipeline()
            
            print("\n运行预处理管道...")
            start_time = time.time()
            pipeline_results = pipeline.run(self.config['data']['raw_path'])
            total_time = time.time() - start_time
            
            results['details']['total_time'] = f"{total_time:.2f}秒"
            results['details']['status'] = pipeline_results['status']
            
            if pipeline_results['status'] == 'completed':
                print(f"\n✓ 管道执行成功")
                print(f"  - 总耗时: {total_time:.2f}秒")
                print(f"  - 执行时间: {pipeline_results['duration']}")
                
                # 显示各步骤结果
                print(f"\n各步骤执行结果:")
                for step, info in pipeline_results['steps'].items():
                    print(f"  - {step}: {info['status']}")
                    
                # 检查输出文件
                output_dir = Path(self.config['preprocessing']['pipeline']['output_dir'])
                output_files = list(output_dir.glob('*.parquet')) + list(output_dir.glob('*.json'))
                
                results['details']['output_files'] = [str(f.name) for f in output_files]
                
                print(f"\n输出文件:")
                for f in output_files:
                    size_mb = f.stat().st_size / 1024 / 1024
                    print(f"  - {f.name} ({size_mb:.2f} MB)")
                    
                results['status'] = 'passed'
                
            else:
                results['status'] = 'failed'
                results['details']['error'] = pipeline_results.get('error')
                print(f"\n✗ 管道执行失败: {pipeline_results.get('error')}")
                
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"\n✗ 测试失败: {e}")
            
        self.results['tests']['pipeline'] = results
        return results
        
    def visualize_results(self, df):
        """可视化数据分析结果"""
        print("\n" + "="*60)
        print("生成数据可视化")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('数据预处理结果分析', fontsize=16)
        
        # 1. 年份分布
        ax1 = axes[0, 0]
        year_counts = df['年份'].value_counts().sort_index()
        ax1.plot(year_counts.index, year_counts.values, marker='o')
        ax1.set_title('文档年份分布')
        ax1.set_xlabel('年份')
        ax1.set_ylabel('文档数量')
        ax1.grid(True, alpha=0.3)
        
        # 2. 文本长度分布
        ax2 = axes[0, 1]
        text_lengths = df['全文长度'].dropna()
        ax2.hist(text_lengths, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_title('文本长度分布')
        ax2.set_xlabel('字符数')
        ax2.set_ylabel('频率')
        ax2.axvline(text_lengths.mean(), color='red', linestyle='--', label=f'平均: {text_lengths.mean():.0f}')
        ax2.legend()
        
        # 3. 分类分布（前10个）
        ax3 = axes[1, 0]
        category_counts = df['分类'].value_counts().head(10)
        ax3.barh(category_counts.index, category_counts.values)
        ax3.set_title('文档分类分布 (Top 10)')
        ax3.set_xlabel('文档数量')
        
        # 4. 数据完整性
        ax4 = axes[1, 1]
        null_counts = df.isnull().sum()
        important_fields = ['年份', '文章名称+副标题', '作者名称', '全文', '分类']
        null_data = null_counts[important_fields]
        colors = ['green' if v == 0 else 'orange' if v < 10 else 'red' for v in null_data.values]
        ax4.bar(null_data.index, null_data.values, color=colors)
        ax4.set_title('重要字段缺失值统计')
        ax4.set_ylabel('缺失值数量')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = Path('data/processed/preprocessing_analysis.png')
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 可视化结果已保存到: {output_path}")
        
        plt.close()
        
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("生成测试报告")
        print("="*60)
        
        # 汇总结果
        summary = {
            'total_tests': len(self.results['tests']),
            'passed': sum(1 for t in self.results['tests'].values() if t.get('status') == 'passed'),
            'failed': sum(1 for t in self.results['tests'].values() if t.get('status') == 'failed'),
            'timestamp': self.results['timestamp']
        }
        
        self.results['summary'] = summary
        
        # 保存JSON报告
        report_path = Path('data/processed/preprocessing_test_report.json')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
            
        print(f"\n测试报告摘要:")
        print(f"  - 总测试数: {summary['total_tests']}")
        print(f"  - 通过: {summary['passed']}")
        print(f"  - 失败: {summary['failed']}")
        print(f"  - 成功率: {summary['passed']/summary['total_tests']*100:.1f}%")
        print(f"\n详细报告已保存到: {report_path}")
        
        return self.results


def main():
    """主测试函数"""
    print("\n🔧 数据预处理模块专项测试")
    print("="*60)
    
    tester = PreprocessingTester()
    
    # 1. 测试Excel解析
    df = tester.test_excel_parser()
    
    # 2. 测试文本处理
    if df is not None:
        tester.test_text_processor(df)
        
    # 3. 测试数据验证
    if df is not None:
        tester.test_data_validator(df)
        
    # 4. 测试完整管道
    tester.test_pipeline()
    
    # 5. 生成可视化（如果有数据）
    if df is not None:
        tester.visualize_results(df)
        
    # 6. 生成测试报告
    report = tester.generate_report()
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)
    
    # 建议后续步骤
    print("\n建议后续步骤:")
    print("1. 查看 data/processed/ 目录下的输出文件")
    print("2. 检查 preprocessing_test_report.json 了解详细结果")
    print("3. 查看 preprocessing_analysis.png 了解数据分布")
    
    if report['summary']['failed'] > 0:
        print("4. ⚠️  有测试失败，请检查错误信息并修复")
    else:
        print("4. ✅ 所有测试通过，可以继续进行RAG索引")


if __name__ == "__main__":
    main()