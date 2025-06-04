"""
test_rag.py

# 运行所有测试
python test_rag.py

# 运行特定测试
python test_rag.py --test search
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import time
import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# 导入RAG组件
from src.rag.engine import RAGEngine
from src.rag.embedding import EmbeddingModel
from src.analysis.concept_analyzer import ConceptAnalyzer
from src.analysis.author_analyzer import AuthorAnalyzer

# 导入统一的模型配置
from model_config import get_embedding_model_config

logger = logging.getLogger(__name__)


class RAGTester:
    """RAG系统测试器"""

    def __init__(self):
        """初始化测试器"""
        # 使用新配置初始化RAG引擎
        self.rag = RAGEngine(
            collection_name="art_design_docs_v2",
            persist_directory="data/chroma_v2",
            embedding_model=None,  # 将单独设置
            enable_bm25=True
        )

        # 设置高维度嵌入模型（使用统一配置）
        embedding_kwargs = get_embedding_model_config(
            model_type='bge-large-zh',
            use_fp16=True,
            batch_size=64
        )
        self.rag.embedding_model = EmbeddingModel(**embedding_kwargs)

        logger.info("初始化RAG系统")
        logger.info(f"嵌入模型: {self.rag.embedding_model.model_name}")
        logger.info(f"向量维度: {self.rag.embedding_model.embedding_dim}")

    def test_basic_search(self):
        """测试基础搜索功能"""
        print("\n" + "=" * 70)
        print("1. 测试基础搜索功能")
        print("=" * 70)

        test_queries = [
            "包豪斯的设计理念是什么",
            "现代主义设计的特点",
            "工业设计和工艺美术的区别",
            "中国传统设计文化"
        ]

        for query in test_queries:
            print(f"\n查询: {query}")
            start_time = time.time()

            results, stats = self.rag.hybrid_search(query, top_k=5)

            elapsed = time.time() - start_time
            print(f"检索时间: {elapsed:.3f}秒")
            print(f"找到 {len(results)} 个结果")

            if results:
                # 显示前3个结果
                for i, doc in enumerate(results[:3], 1):
                    metadata = doc.get('metadata', {})
                    score = doc.get('rerank_score', doc.get('score', 0))

                    print(f"\n  [{i}] 相关度: {score:.3f}")
                    print(f"      标题: {metadata.get('文章名称+副标题', 'N/A')}")
                    print(f"      作者: {metadata.get('作者名称', 'N/A')}")
                    print(f"      年份: {metadata.get('年份', 'N/A')}")
                    print(f"      策略: {metadata.get('chunk_strategy', 'N/A')}")
                    print(f"      预览: {doc.get('text', '')[:100]}...")

    def test_concept_analysis(self):
        """测试概念分析功能"""
        print("\n" + "=" * 70)
        print("2. 测试概念分析功能")
        print("=" * 70)

        analyzer = ConceptAnalyzer(self.rag)

        test_concepts = ["包豪斯", "现代主义", "工业设计"]

        for concept in test_concepts:
            print(f"\n分析概念: {concept}")

            # 测试首次出现
            start_time = time.time()
            first_appearance = analyzer.find_first_appearance(concept)
            elapsed = time.time() - start_time

            if first_appearance['status'] == 'found':
                print(f"  首次出现: {first_appearance['year']}年")
                print(f"  文献: 《{first_appearance['title']}》")
                print(f"  作者: {first_appearance['author']}")
                print(f"  分析时间: {elapsed:.3f}秒")
            else:
                print(f"  未找到概念首次出现")

            # 测试时间分布
            temporal_df = analyzer.analyze_temporal_distribution(concept)
            if not temporal_df.empty:
                print(f"  时间分布: {len(temporal_df)}个年份")
                print(f"  总出现次数: {temporal_df['count'].sum()}")

    # def test_performance_comparison(self):
    #     """性能对比测试"""
    #     print("\n" + "=" * 70)
    #     print("3. 性能对比测试")
    #     print("=" * 70)
    #
    #     # 准备测试查询
    #     test_queries = [
    #         "包豪斯学校的建立和影响",
    #         "装饰艺术运动的特征",
    #         "极简主义设计理念",
    #         "后现代主义的批判性",
    #         "数字化设计的发展趋势"
    #     ]
    #
    #     # 测试不同检索方法
    #     methods = ['vector', 'bm25', 'hybrid']
    #     results_summary = {method: {'time': [], 'count': [], 'scores': []} for method in methods}
    #
    #
    #     for query in test_queries:
    #         print(f"\n查询: {query}")
    #
    #         for method in methods:
    #             start_time = time.time()
    #             results, stats = self.rag.hybrid_search(
    #                 query,
    #                 top_k=10,
    #                 method=method
    #             )
    #             elapsed = time.time() - start_time
    #
    #             # 记录结果
    #             results_summary[method]['time'].append(elapsed)
    #             results_summary[method]['count'].append(len(results))
    #             if results:
    #                 avg_score = np.mean([r.get('score', 0) for r in results[:5]])
    #                 results_summary[method]['scores'].append(avg_score)
    #             else:
    #                 results_summary[method]['scores'].append(0)
    #
    #             print(f"  {method}: {elapsed:.3f}秒, {len(results)}个结果")
    #
    #     # 打印汇总
    #     print("\n性能汇总:")
    #     print(f"{'方法':<10} {'平均时间':<10} {'平均结果数':<12} {'平均相关度':<10}")
    #     print("-" * 45)
    #
    #     for method in methods:
    #         avg_time = np.mean(results_summary[method]['time'])
    #         avg_count = np.mean(results_summary[method]['count'])
    #         avg_score = np.mean(results_summary[method]['scores'])
    #         print(f"{method:<10} {avg_time:<10.3f} {avg_count:<12.1f} {avg_score:<10.3f}")

    def test_performance_comparison(self):
        """性能对比测试 - 修复版本"""
        print("\n" + "=" * 70)
        print("3. 性能对比测试")
        print("=" * 70)

        # 准备测试查询
        test_queries = [
            "包豪斯学校的建立和影响",
            "装饰艺术运动的特征",
            "极简主义设计理念",
            "后现代主义的批判性",
            "数字化设计的发展趋势"
        ]

        # 测试不同检索方法
        methods = ['vector', 'bm25', 'hybrid']
        results_summary = {method: {'time': [], 'count': [], 'scores': []} for method in methods}

        for query in test_queries:
            print(f"\n查询: {query}")

            for method in methods:
                start_time = time.time()
                results, stats = self.rag.hybrid_search(
                    query=query,
                    top_k=10,
                    method=method
                )
                elapsed = time.time() - start_time

                # 记录结果
                results_summary[method]['time'].append(elapsed)
                results_summary[method]['count'].append(len(results))

                if results:
                    # 🔧 修复：手动归一化分数进行公平比较
                    raw_scores = []

                    # 提取原始分数
                    for r in results[:5]:
                        if method == 'vector':
                            # 向量检索：使用distance转换为相似度，或直接使用score
                            if 'distance' in r:
                                score = 1 - r['distance']  # distance越小，相似度越高
                            else:
                                score = r.get('score', 0)
                        elif method == 'bm25':
                            # BM25检索：使用原始BM25分数
                            score = r.get('score', 0)
                        else:  # hybrid
                            # 混合检索：优先使用combined_score
                            score = r.get('combined_score', r.get('rerank_score', r.get('score', 0)))

                        raw_scores.append(score)

                    # 手动归一化到[0,1]范围
                    if raw_scores:
                        min_score = min(raw_scores)
                        max_score = max(raw_scores)

                        if max_score > min_score:
                            # Min-Max归一化
                            normalized_scores = [(s - min_score) / (max_score - min_score) for s in raw_scores]
                        else:
                            # 所有分数相同的情况
                            normalized_scores = [0.5] * len(raw_scores)

                        avg_score = np.mean(normalized_scores)
                    else:
                        avg_score = 0

                    results_summary[method]['scores'].append(avg_score)

                    # 调试输出：显示原始分数范围
                    if raw_scores:
                        print(f"    {method}: {elapsed:.3f}秒, {len(results)}个结果 "
                              f"(分数范围: {min(raw_scores):.3f}-{max(raw_scores):.3f})")
                    else:
                        print(f"    {method}: {elapsed:.3f}秒, {len(results)}个结果")
                else:
                    results_summary[method]['scores'].append(0)
                    print(f"    {method}: {elapsed:.3f}秒, 0个结果")

        # 打印汇总统计
        print("\n性能汇总:")
        print(f"{'方法':<10} {'平均时间':<10} {'平均结果数':<12} {'平均相关度':<12} {'时间排名':<8}")
        print("-" * 55)

        # 计算排名
        avg_times = {method: np.mean(results_summary[method]['time']) for method in methods}
        time_ranking = sorted(avg_times.items(), key=lambda x: x[1])
        time_ranks = {method: i + 1 for i, (method, _) in enumerate(time_ranking)}

        for method in methods:
            avg_time = np.mean(results_summary[method]['time'])
            avg_count = np.mean(results_summary[method]['count'])
            avg_score = np.mean(results_summary[method]['scores'])
            time_rank = time_ranks[method]

            print(f"{method:<10} {avg_time:<10.3f} {avg_count:<12.1f} {avg_score:<12.3f} #{time_rank}")

        # 额外的分析
        print(f"\n📊 详细分析:")

        # 最快的方法
        fastest_method = min(avg_times.items(), key=lambda x: x[1])
        print(f"⚡ 最快检索: {fastest_method[0]} ({fastest_method[1]:.3f}秒)")

        # 最高相关度的方法
        avg_relevance = {method: np.mean(results_summary[method]['scores']) for method in methods}
        best_relevance = max(avg_relevance.items(), key=lambda x: x[1])
        print(f"🎯 最高相关度: {best_relevance[0]} ({best_relevance[1]:.3f})")

        # 混合检索的效率分析
        if 'hybrid' in avg_times and 'vector' in avg_times:
            hybrid_overhead = avg_times['hybrid'] - avg_times['vector']
            print(f"🔄 混合检索开销: +{hybrid_overhead:.3f}秒 ({hybrid_overhead / avg_times['vector'] * 100:.1f}%)")

        # 分数一致性检查
        print(f"\n🔍 分数分布检查:")
        for method in methods:
            scores = results_summary[method]['scores']
            if scores:
                std_dev = np.std(scores)
                print(
                    f"  {method}: 标准差={std_dev:.3f} (一致性: {'高' if std_dev < 0.1 else '中' if std_dev < 0.2 else '低'})")

    def test_chunk_strategies(self):
        """测试不同chunk策略的效果"""
        print("\n" + "=" * 70)
        print("4. 测试Chunk策略效果")
        print("=" * 70)

        # 这个测试需要分别用不同策略构建的数据
        # 这里展示如何分析当前策略的效果

        test_query = "包豪斯的教育理念和方法"
        results, _ = self.rag.hybrid_search(test_query, top_k=10)

        # 统计不同策略的结果
        strategy_counts = {}
        for doc in results:
            strategy = doc.get('metadata', {}).get('chunk_strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        print(f"查询: {test_query}")
        print(f"结果中的chunk策略分布:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count}个")

        # 分析chunk长度与相关度的关系
        if results:
            lengths = []
            scores = []
            for doc in results:
                length = int(doc.get('metadata', {}).get('chunk_length', 0))
                # 使用适当的分数字段
                score = doc.get('rerank_score',
                                doc.get('combined_score', doc.get('normalized_score', doc.get('score', 0))))
                lengths.append(length)
                scores.append(score)

            if lengths and scores:
                correlation = np.corrcoef(lengths, scores)[0, 1]
                print(f"\nChunk长度与相关度的相关系数: {correlation:.3f}")
                print(f"平均chunk长度: {np.mean(lengths):.0f}字符")
                print(f"平均相关度分数: {np.mean(scores):.3f}")

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 70)
        print("开始测试RAG系统")
        print("=" * 70)

        # 检查向量库状态
        doc_count = self.rag.vector_store.count()
        print(f"\n向量库文档数: {doc_count}")

        if doc_count == 0:
            print("❌ 向量库为空，请先运行 build_rag.py 构建数据")
            return

        # 运行各项测试
        self.test_basic_search()
        self.test_concept_analysis()
        self.test_performance_comparison()
        self.test_chunk_strategies()

        print("\n" + "=" * 70)
        print("✅ 测试完成！")
        print("=" * 70)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='测试RAG系统')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'search', 'concept', 'performance', 'chunk'],
                        help='要运行的测试类型')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建测试器
    tester = RAGTester()

    # 运行测试
    if args.test == 'all':
        tester.run_all_tests()
    elif args.test == 'search':
        tester.test_basic_search()
    elif args.test == 'concept':
        tester.test_concept_analysis()
    elif args.test == 'performance':
        tester.test_performance_comparison()
    elif args.test == 'chunk':
        tester.test_chunk_strategies()


if __name__ == "__main__":
    main()