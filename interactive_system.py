"""
interactive_system.py
快速启动的RAG交互系统 - 适配1024维度高质量中文模型
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
from typing import List, Dict, Optional, Any, Tuple

import pandas as pd
import yaml

# 导入核心模块
from src.rag.engine import RAGEngine
from src.rag.embedding import EmbeddingModel

# 导入分析模块 - 使用优化版本
from src.analysis.author_analyzer import AuthorAnalyzer
from src.analysis.concept_analyzer import ConceptAnalyzer


class FastRAGInteractiveSystem:
    """快速启动的RAG交互系统 - 1024维模型优化版"""

    def __init__(self):
        """初始化系统"""
        self.print_header()
        self.load_config()
        self.init_embedding_model()
        self.init_rag_engine()
        self.check_database_status()

        # 延迟初始化分析器
        self.author_analyzer = None
        self.concept_analyzer = None

        print("\n✅ 系统启动完成！")

    def print_header(self):
        """打印系统标题"""
        print("=" * 70)
        print("🎨 艺术设计RAG系统 v2.0 - 高精度版")
        print("=" * 70)
        print("核心功能：")
        print("  ✓ 1024维高质量中文嵌入模型")
        print("  ✓ 智能文档检索 (混合算法)")
        print("  ✓ 作者搜索与分析（延迟加载）")
        print("  ✓ 概念分析与演进（延迟加载）")
        print("  ✓ 优化的内存管理")
        print("=" * 70)

    def load_config(self):
        """加载配置"""
        try:
            # 尝试多个配置文件
            config_files = ['config.yaml', 'config.yaml.backup']
            self.config = {}

            for config_path in config_files:
                if Path(config_path).exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self.config = yaml.safe_load(f)
                    print(f"✓ 加载配置文件: {config_path}")
                    break

            if not self.config:
                print("⚠️  未找到配置文件，使用默认配置")
                self.config = self.get_default_config()

        except Exception as e:
            print(f"⚠️  加载配置失败: {e}，使用默认配置")
            self.config = self.get_default_config()

    def get_default_config(self):
        """获取默认配置"""
        return {
            'embedding': {
                'current_model': 'bge_large',
                'models': {
                    'bge_large': {
                        'model_name': 'BAAI/bge-large-zh-v1.5',
                        'dimension': 1024,
                        'device': 'cuda'
                    }
                },
                'mixed_precision': True,
                'batch_size': 32
            },
            'vector_store': {
                'collection_name': 'art_design_docs_v2',
                'persist_directory': 'data/chroma_v2'
            },
            'retrieval': {
                'default_top_k': 5,
                'bm25_weight': 0.3,
                'vector_weight': 0.7
            }
        }

    def init_embedding_model(self):
        """初始化高质量嵌入模型"""
        print("\n🤖 初始化嵌入模型...")

        # 从配置获取模型信息
        embedding_config = self.config.get('embedding', {})
        current_model = embedding_config.get('current_model', 'bge_large')
        models_config = embedding_config.get('models', {})

        if current_model not in models_config:
            print("⚠️  配置的模型不存在，使用默认BGE-Large模型")
            current_model = 'bge_large'
            models_config = self.get_default_config()['embedding']['models']

        model_config = models_config[current_model]

        try:
            self.embedding_model = EmbeddingModel(
                model_name=model_config['model_name'],
                device=model_config.get('device', 'cuda'),
                use_fp16=embedding_config.get('mixed_precision', True),
                batch_size=embedding_config.get('batch_size', 32)
            )

            print(f"✓ 嵌入模型: {model_config['model_name']}")
            print(f"✓ 模型维度: {self.embedding_model.embedding_dim}")
            print(f"✓ 设备: {self.embedding_model.device}")
            print(f"✓ 批次大小: {self.embedding_model.batch_size}")

            # 验证维度
            if self.embedding_model.embedding_dim != model_config['dimension']:
                print(f"⚠️  实际维度({self.embedding_model.embedding_dim}) != 配置维度({model_config['dimension']})")

        except Exception as e:
            print(f"❌ 嵌入模型初始化失败: {e}")
            print("回退到默认模型...")
            self.embedding_model = EmbeddingModel(model_type='bge-large-zh')

    def init_rag_engine(self):
        """初始化RAG引擎"""
        print("\n🔧 初始化RAG引擎...")

        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

        # 获取向量存储配置
        vector_config = self.config.get('vector_store', {})
        collection_name = vector_config.get('collection_name', 'art_design_docs_v2')
        persist_directory = vector_config.get('persist_directory', 'data/chroma_v2')

        # 使用预初始化的嵌入模型创建RAG引擎
        self.rag = RAGEngine(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=None,  # 稍后手动设置
            gemini_api_key=api_key,
            enable_bm25=True
        )

        # 手动设置嵌入模型
        self.rag.embedding_model = self.embedding_model

        # 重新初始化检索器以使用新的嵌入模型
        self.rag.retriever.embedding_model = self.embedding_model

        print(f"✓ RAG引擎初始化完成")
        print(f"✓ 集合名称: {collection_name}")
        print(f"✓ 存储目录: {persist_directory}")

        if api_key:
            print("✓ Gemini API已配置")
        else:
            print("⚠️  未配置Gemini API（可选）")

    def check_database_status(self):
        """检查数据库状态"""
        print("\n🗄️  检查数据库状态...")

        doc_count = self.rag.vector_store.count()

        if doc_count == 0:
            print("\n❌ 向量数据库为空！")
            print("请先运行以下命令构建索引：")
            print("  python build_rag.py --input data/raw/applied_arts.xlsx")
            print("  或")
            print("  python test_preprocessing.py")

            # 询问是否继续
            choice = input("\n是否继续启动系统？(y/n): ").strip().lower()
            if choice != 'y':
                sys.exit(1)
            else:
                print("⚠️  系统将以有限功能启动")
        else:
            print(f"✓ 向量数据库就绪（{doc_count:,}个文档）")

        # 检查BM25索引
        if hasattr(self.rag, 'retriever') and self.rag.retriever.bm25_index:
            bm25_docs = len(self.rag.retriever.documents)
            print(f"✓ BM25索引就绪（{bm25_docs:,}个文档）")
        else:
            print("⚠️  BM25索引未构建（将影响混合检索效果）")

            if doc_count > 0:
                print("💡 提示: 首次使用混合检索时将自动构建BM25索引")

    def init_author_analyzer(self):
        """延迟初始化作者分析器"""
        if self.author_analyzer is None:
            print("\n🔧 首次使用作者功能，初始化分析器...")
            start_time = time.time()

            # 使用延迟加载的优化版本
            self.author_analyzer = AuthorAnalyzer(self.rag, lazy_load=True)

            elapsed = time.time() - start_time
            print(f"✓ 作者分析器就绪 ({elapsed:.1f}秒)")

    def init_concept_analyzer(self):
        """延迟初始化概念分析器"""
        if self.concept_analyzer is None:
            print("\n🔧 首次使用概念功能，初始化分析器...")
            start_time = time.time()

            self.concept_analyzer = ConceptAnalyzer(self.rag)

            elapsed = time.time() - start_time
            print(f"✓ 概念分析器就绪 ({elapsed:.1f}秒)")

    def run(self):
        """运行主循环"""
        print("\n" + "=" * 70)
        print("🚀 系统就绪！输入 'help' 查看命令列表")
        print("💡 提示: 首次搜索可能需要几秒钟构建索引")
        print("=" * 70)

        while True:
            try:
                query = input("\n🔍 输入> ").strip()

                if not query:
                    continue

                # 处理退出命令
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 感谢使用，再见！")
                    break

                # 处理帮助命令
                if query.lower() in ['help', 'h', '?']:
                    self.show_help()
                    continue

                # 处理系统命令
                if query.lower() == 'stats':
                    self.show_system_stats()
                    continue

                if query.lower() == 'authors':
                    self.show_author_list()
                    continue

                if query.lower() == 'test':
                    self.run_tests()
                    continue

                if query.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    self.print_header()
                    continue

                # 处理分析命令
                if query.startswith('/'):
                    self.handle_command(query)
                    continue

                # 默认：普通搜索
                self.handle_search(query)

            except KeyboardInterrupt:
                print("\n\n使用 'exit' 退出程序")
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                # 在调试模式下显示详细错误
                if os.getenv('DEBUG'):
                    import traceback
                    traceback.print_exc()

    def show_help(self):
        """显示帮助信息"""
        print("\n" + "=" * 70)
        print("📚 命令帮助")
        print("=" * 70)

        print("\n🔍 基础搜索:")
        print("  直接输入关键词进行文档搜索")
        print("  例如: 包豪斯设计理念")

        print("\n👤 作者分析命令:")
        print("  /author <作者名>        - 分析指定作者的所有文章")
        print("  /find_author <关键词>   - 搜索包含关键词的作者")

        print("\n💡 概念分析命令:")
        print("  /concept <概念>         - 分析概念的详细信息")
        print("  /first <概念>           - 查找概念的首次出现")
        print("  /related <概念>         - 查找相关概念")
        print("  /evolution <概念>       - 分析概念演进")

        print("\n⚙️  系统命令:")
        print("  stats                   - 显示系统统计信息")
        print("  authors                 - 显示作者列表")
        print("  test                    - 运行功能测试")
        print("  clear                   - 清屏")
        print("  help                    - 显示此帮助")
        print("  exit                    - 退出程序")

        print("\n💡 搜索技巧:")
        print("  • 使用具体的概念词汇可以获得更精确的结果")
        print("  • 1024维模型对中文语义理解更准确")
        print("  • 混合检索结合了语义和关键词匹配")

    def handle_command(self, command: str):
        """处理命令"""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        print(f"\n🎯 执行命令: {cmd}")

        # 作者相关命令
        if cmd == '/author':
            if args:
                self.init_author_analyzer()  # 延迟初始化
                self.analyze_author(args)
            else:
                print("❌ 请指定作者名: /author 作者名")

        elif cmd == '/find_author':
            if args:
                self.init_author_analyzer()  # 延迟初始化
                self.find_authors(args)
            else:
                print("❌ 请指定搜索关键词: /find_author 关键词")

        # 概念相关命令
        elif cmd == '/concept':
            if args:
                self.init_concept_analyzer()  # 延迟初始化
                self.analyze_concept(args)
            else:
                print("❌ 请指定概念: /concept 概念名")

        elif cmd == '/first':
            if args:
                self.init_concept_analyzer()  # 延迟初始化
                self.find_first_appearance(args)
            else:
                print("❌ 请指定概念: /first 概念名")

        elif cmd == '/related':
            if args:
                self.init_concept_analyzer()  # 延迟初始化
                self.find_related_concepts(args)
            else:
                print("❌ 请指定概念: /related 概念名")

        elif cmd == '/evolution':
            if args:
                self.init_concept_analyzer()  # 延迟初始化
                self.analyze_concept_evolution(args)
            else:
                print("❌ 请指定概念: /evolution 概念名")

        else:
            print(f"❌ 未知命令: {cmd}")
            print("输入 'help' 查看可用命令")

    def handle_search(self, query: str):
        """处理普通搜索 - 优化版"""
        print(f"🔍 搜索: {query}")

        # 获取配置的检索参数
        retrieval_config = self.config.get('retrieval', {})
        top_k = retrieval_config.get('default_top_k', 5)
        bm25_weight = retrieval_config.get('bm25_weight', 0.3)
        vector_weight = retrieval_config.get('vector_weight', 0.7)

        start_time = time.time()

        try:
            results, stats = self.rag.hybrid_search(
                query=query,
                top_k=top_k,
                method="hybrid",
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
                rerank=True
            )

            elapsed = time.time() - start_time

            print(f"⏱️  搜索完成 ({elapsed:.2f}秒)")
            print(f"   方法: {stats.get('method', 'hybrid')}")
            print(f"   结果数: {len(results)}")

            # 显示检索统计
            if 'vector_results' in stats:
                print(f"   向量检索: {stats['vector_results']}个")
            if 'bm25_results' in stats:
                print(f"   BM25检索: {stats['bm25_results']}个")

            if not results:
                print("❌ 未找到相关文档")
                print("💡 尝试使用更通用的关键词或检查拼写")
                return

            print("\n" + "=" * 60)
            for i, doc in enumerate(results, 1):
                self._display_search_result(i, doc)

            # 询问是否生成AI答案
            if self.rag.generator.is_available() and len(results) > 0:
                print("\n" + "-" * 60)
                choice = input("💡 是否需要AI生成综合答案？(y/n): ").strip().lower()
                if choice == 'y':
                    self._generate_ai_answer(query, results[:3])

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ 搜索失败 ({elapsed:.2f}秒): {e}")

            # 提供故障排除建议
            if "ChromaDB" in str(e) or "collection" in str(e):
                print("💡 可能是数据库问题，请检查向量库是否正确构建")
            elif "CUDA" in str(e) or "memory" in str(e):
                print("💡 可能是GPU内存不足，可尝试减小批次大小")

    def _display_search_result(self, rank: int, doc: Dict):
        """显示搜索结果 - 增强版"""
        metadata = doc.get('metadata', {})

        # 获取各种分数
        scores = []
        if 'rerank_score' in doc:
            scores.append(f"重排序: {doc['rerank_score']:.3f}")
        if 'combined_score' in doc:
            scores.append(f"综合: {doc['combined_score']:.3f}")
        if 'score' in doc:
            scores.append(f"原始: {doc['score']:.3f}")

        score_str = " | ".join(scores) if scores else "N/A"

        print(f"\n【{rank}】{metadata.get('文章名称+副标题', '无标题')}")
        print(f"   📅 年份: {metadata.get('年份', 'N/A')}")
        print(f"   👤 作者: {metadata.get('作者名称', 'N/A')}")
        print(f"   📂 分类: {metadata.get('分类', 'N/A')}")
        print(f"   📊 相关度: {score_str}")

        # 显示检索方法（如果有）
        if 'retrieval_method' in doc:
            print(f"   🔍 检索方式: {doc['retrieval_method']}")

        # 显示文本片段
        text = doc.get('text', '')[:300]
        print(f"   📝 摘要: {text}...")

        # 显示chunk信息（如果有）
        if 'chunk_strategy' in metadata:
            print(f"   🧩 块策略: {metadata['chunk_strategy']}")

    # def _generate_ai_answer(self, query: str, context: List[Dict]):
    #     """生成AI答案"""
    #     print("\n🤖 正在生成AI答案...")
    #
    #     try:
    #         answer = self.rag.generate_answer_with_citations(query, context)
    #         print("\n" + "=" * 60)
    #         print("💡 AI综合答案:")
    #         print("=" * 60)
    #         print(answer)
    #         print("=" * 60)
    #     except Exception as e:
    #         print(f"❌ 生成答案失败: {e}")
    #         if "API" in str(e):
    #             print("💡 请检查API密钥配置")

    def _generate_ai_answer(self, query: str, context: List[Dict]):
        """生成AI答案 - 修复版本"""
        print("\n🤖 正在生成AI答案...")

        try:
            # 方法1：使用 generate_with_citations
            answer = self.rag.generate_answer_with_citations(query, context)

            print("\n" + "=" * 60)
            print("💡 AI综合答案:")
            print("=" * 60)
            print(answer)
            print("=" * 60)

        except Exception as e:
            # 详细的错误处理
            error_msg = str(e)
            print(f"❌ 生成答案失败: {error_msg}")

            # 针对特定错误提供解决方案
            if "Invalid format specifier" in error_msg:
                print("💡 检测到格式化错误，尝试备用方法...")

                # 备用方法：直接调用基础 generate 方法
                try:
                    # 使用更简单的提示词，避免复杂格式
                    simple_prompt = """你是艺术设计领域的专家。请基于提供的文档回答问题。

    重要规则：
    1. 只使用文档中的信息
    2. 保持专业和准确
    3. 如果信息不足，请说明"""

                    answer = self.rag.generator.generate(
                        query=query,
                        context=context,
                        system_prompt=simple_prompt
                    )

                    print("\n" + "=" * 60)
                    print("💡 AI综合答案（简化版）:")
                    print("=" * 60)
                    print(answer)
                    print("=" * 60)

                except Exception as e2:
                    print(f"❌ 备用方法也失败: {e2}")
                    self._show_manual_summary(query, context)

            elif "API" in error_msg:
                print("💡 请检查API密钥配置")
                print("   设置环境变量: export GOOGLE_API_KEY='your_key'")
            else:
                # 显示更多调试信息
                if os.getenv('DEBUG'):
                    import traceback
                    traceback.print_exc()

                # 提供手动摘要
                self._show_manual_summary(query, context)

    def _show_manual_summary(self, query: str, context: List[Dict]):
        """显示手动摘要作为后备方案"""
        print("\n" + "=" * 60)
        print("📋 相关文档摘要（手动整理）:")
        print("=" * 60)

        print(f"关于「{query}」的相关信息：\n")

        for i, doc in enumerate(context[:3], 1):
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')

            print(f"{i}. 《{metadata.get('文章名称+副标题', '无标题')}》")
            print(f"   作者：{metadata.get('作者名称', '未知')}")
            print(f"   年份：{metadata.get('年份', '未知')}")
            print(f"   相关内容：")

            # 提取关键句子
            sentences = text.split('。')
            relevant_sentences = []

            # 简单的相关性判断
            query_words = set(query.split())
            for sent in sentences[:10]:  # 只看前10个句子
                if any(word in sent for word in query_words):
                    relevant_sentences.append(sent.strip())

            if relevant_sentences:
                for sent in relevant_sentences[:3]:
                    print(f"   • {sent}。")
            else:
                print(f"   • {text[:200]}...")

            print()

        print("=" * 60)
        print("💡 提示：这是基于文档的简单摘要。启用AI功能可获得更好的综合答案。")

    def analyze_author(self, author_name: str):
        """分析作者"""
        print(f"📖 分析作者: {author_name}")

        start_time = time.time()
        result = self.author_analyzer.analyze_author(author_name)
        elapsed = time.time() - start_time

        print(f"⏱️  分析完成 ({elapsed:.2f}秒)")

        if result['status'] == 'not_found':
            print(f"❌ {result['message']}")

            # 提供相似作者建议
            similar = self.author_analyzer.search_authors(author_name)
            if similar:
                print(f"\n💡 是否要查找以下相似作者？")
                for i, (author, count) in enumerate(similar[:5], 1):
                    print(f"   {i}. {author} ({count}篇)")
            return

        # 显示分析结果
        print(f"\n✅ 找到作者: {result['matched_author']}")
        if result['query_author'] != result['matched_author']:
            print(f"   (查询: {result['query_author']})")

        print(f"   🎯 匹配度: {result['match_confidence']:.2%}")
        print(f"   📚 发文总数: {result['total_publications']}")

        # 显示发表文章
        print(f"\n📄 发表文章:")
        for i, pub in enumerate(result['publications'][:15], 1):
            year_str = f"[{pub['year']}]" if pub['year'] else "[未知]"
            category_str = f" - {pub['category']}" if pub['category'] else ""
            print(f"   {i:2d}. {year_str} {pub['title']}{category_str}")

        if len(result['publications']) > 15:
            print(f"   ... 还有 {len(result['publications']) - 15} 篇文章")

        # 显示年份分布
        if result['year_distribution']:
            print(f"\n📊 年份分布:")
            sorted_years = sorted(result['year_distribution'].items())
            for year, count in sorted_years:
                bar = '█' * min(count, 20)
                print(f"   {year}: {bar} ({count}篇)")

        # 显示分类分布
        if result['category_distribution']:
            print(f"\n📚 分类分布:")
            for category, count in list(result['category_distribution'].items())[:5]:
                print(f"   {category}: {count}篇")

        # 显示合作者
        if result['collaborators']:
            print(f"\n👥 主要合作者:")
            for collaborator, count in list(result['collaborators'].items())[:8]:
                print(f"   • {collaborator} ({count}次)")

        # 显示研究主题
        if result['research_topics']:
            print(f"\n🎯 研究主题:")
            for topic, count in list(result['research_topics'].items())[:8]:
                print(f"   • {topic} ({count}次)")

    def find_authors(self, pattern: str):
        """搜索作者"""
        print(f"🔍 搜索作者: {pattern}")

        matches = self.author_analyzer.search_authors(pattern)

        if not matches:
            print(f"❌ 未找到匹配 '{pattern}' 的作者")
            return

        print(f"\n✅ 找到 {len(matches)} 个匹配的作者:")
        for i, (author, count) in enumerate(matches[:20], 1):
            print(f"   {i:2d}. {author:<15} ({count}篇)")

        if len(matches) > 20:
            print(f"   ... 还有 {len(matches) - 20} 个作者")

    def analyze_concept(self, concept: str):
        """分析概念"""
        print(f"💡 分析概念: {concept}")

        start_time = time.time()
        report = self.concept_analyzer.generate_concept_report(concept)
        elapsed = time.time() - start_time

        print(f"⏱️  分析完成 ({elapsed:.2f}秒)")

        if report['status'] == 'not_found':
            print(f"❌ 未找到概念 '{concept}' 的相关文档")
            return

        # 显示首次出现
        first = report.get('first_appearance', {})
        if first.get('status') == 'found':
            print(f"\n📍 首次出现:")
            print(f"   年份: {first['year']}")
            print(f"   文献: 《{first['title']}》")
            print(f"   作者: {first['author']}")
            print(f"   找到文档: {first['total_docs_found']}篇")

        # 显示统计信息
        stats = report.get('statistics', {})
        if stats:
            print(f"\n📊 统计信息:")
            print(f"   总提及次数: {stats.get('total_mentions', 0)}")
            print(f"   活跃年份数: {stats.get('active_years', 0)}")
            print(f"   年份范围: {stats.get('year_range', 'N/A')}")
            print(f"   峰值年份: {stats.get('peak_year', 'N/A')} ({stats.get('peak_count', 0)}篇)")

        # 显示相关概念
        related = report.get('related_concepts', [])
        if related:
            print(f"\n🔗 相关概念:")
            for concept_name, count in related[:8]:
                print(f"   • {concept_name} (共现{count}次)")

    def find_first_appearance(self, concept: str):
        """查找概念首次出现"""
        print(f"📍 查找概念首次出现: {concept}")

        result = self.concept_analyzer.find_first_appearance(concept)

        if result['status'] == 'found':
            print(f"\n✅ 找到首次出现:")
            print(f"   年份: {result['year']}")
            print(f"   文献: 《{result['title']}》")
            print(f"   作者: {result['author']}")
            print(f"   分类: {result.get('category', '未知')}")
            print(f"   找到文档: {result['total_docs_found']}篇")
            print(f"\n📝 上下文:")
            print(f"   {result['context']}")

        elif result['status'] == 'no_valid_date':
            print(f"⚠️  {result['message']}")

        else:
            print(f"❌ {result['message']}")

    def find_related_concepts(self, concept: str):
        """查找相关概念"""
        print(f"🔗 查找相关概念: {concept}")

        related = self.concept_analyzer.find_related_concepts(concept, top_n=15)

        if not related:
            print(f"❌ 未找到与 '{concept}' 相关的概念")
            return

        print(f"\n✅ 找到 {len(related)} 个相关概念:")
        for i, (related_concept, count) in enumerate(related, 1):
            print(f"   {i:2d}. {related_concept:<15} (共现{count}次)")

    def analyze_concept_evolution(self, concept: str):
        """分析概念演进"""
        print(f"📈 分析概念演进: {concept}")

        evolution = self.concept_analyzer.analyze_concept_evolution(concept)

        if evolution['status'] == 'no_data':
            print(f"❌ {evolution['message']}")
            return

        overview = evolution['overview']
        print(f"\n✅ 演进概览:")
        print(f"   时间跨度: {overview['first_year']}-{overview['last_year']} ({overview['span_years']}年)")
        print(f"   总提及数: {overview['total_mentions']}")
        print(f"   峰值年份: {overview['peak_year']} ({overview['peak_count']}篇)")
        print(f"   整体趋势: {overview['trend']}")

        periods = evolution.get('periods', [])
        if periods:
            print(f"\n📊 分期分析:")
            for period in periods:
                print(f"   {period['period']}: {period['total_mentions']}篇 "
                      f"(峰值: {period['peak_year']})")

    def show_system_stats(self):
        """显示系统统计"""
        print("\n" + "=" * 60)
        print("📊 系统统计")
        print("=" * 60)

        # RAG统计
        rag_stats = self.rag.get_statistics()

        print(f"\n🔍 检索系统:")
        print(f"  向量数据库: {rag_stats['collection_name']}")
        print(f"  文档总数: {rag_stats['total_documents']:,}")
        print(f"  嵌入模型: {rag_stats['embedding_model']}")
        print(f"  嵌入维度: {rag_stats['embedding_dim']}")
        print(f"  BM25启用: {'是' if rag_stats['bm25_enabled'] else '否'}")

        if rag_stats.get('bm25_indexed'):
            print(f"  BM25文档数: {rag_stats.get('bm25_documents', 0):,}")

        # 模型性能信息
        print(f"\n🤖 模型信息:")
        print(f"  设备: {self.embedding_model.device}")
        print(f"  批次大小: {self.embedding_model.batch_size}")
        print(f"  FP16模式: {'开启' if hasattr(self.embedding_model, 'model') and self.embedding_model.device == 'cuda' else '关闭'}")

        # 作者统计（如果已初始化）
        if self.author_analyzer:
            author_stats = self.author_analyzer.get_statistics()
            print(f"\n👤 作者分析:")
            print(f"  索引状态: {'已构建' if author_stats.get('index_built', False) else '未构建'}")
            if author_stats.get('index_built', False):
                print(f"  作者总数: {author_stats['total_authors']:,}")
                print(f"  平均每作者文档数: {author_stats['avg_docs_per_author']:.1f}")
                print(f"  最多文档作者: {author_stats['max_docs_by_author']}篇")
                print(f"  多篇作者数: {author_stats['authors_with_multiple_docs']:,}")
        else:
            print(f"\n👤 作者分析: 未初始化")

        # 概念统计（如果已初始化）
        if self.concept_analyzer:
            concept_stats = self.concept_analyzer.get_concept_statistics()
            print(f"\n💡 概念分析:")
            print(f"  预定义概念: {concept_stats['known_concepts_count']}个")
            print(f"  缓存大小: {concept_stats['cache_size']}")
        else:
            print(f"\n💡 概念分析: 未初始化")

        # 搜索历史
        if hasattr(self.rag, 'search_history') and self.rag.search_history:
            print(f"\n📈 使用统计:")
            print(f"  总查询数: {len(self.rag.search_history)}")

            # 统计查询方法
            methods = {}
            for record in self.rag.search_history:
                method = record.get('method', 'unknown')
                methods[method] = methods.get(method, 0) + 1

            for method, count in methods.items():
                print(f"  {method}: {count}次")

    def show_author_list(self):
        """显示作者列表"""
        print("\n📖 作者列表（按发文数排序，前20位）:")

        # 初始化作者分析器（如果还没有）
        self.init_author_analyzer()

        author_list = self.author_analyzer.get_author_list(limit=20)

        if not author_list:
            print("❌ 暂无作者数据")
            return

        print("-" * 50)
        for i, (author, count) in enumerate(author_list, 1):
            print(f"   {i:2d}. {author:<20} ({count}篇)")

        # 获取统计信息
        stats = self.author_analyzer.get_statistics()
        total_authors = stats.get('total_authors', 0)

        if total_authors > 20:
            print(f"\n   ... 还有 {total_authors - 20:,} 个作者")

        print(f"\n💡 使用 '/author 作者名' 查看具体作者信息")

    def run_tests(self):
        """运行功能测试"""
        print("\n🧪 运行系统测试...")

        # 测试基础搜索
        print("\n1. 测试基础搜索功能:")
        test_query = "设计"
        try:
            results, stats = self.rag.hybrid_search(test_query, top_k=3)
            print(f"   测试查询: {test_query}")
            print(f"   ✅ 成功 - 找到 {len(results)} 个结果")
            if results:
                print(f"   📊 平均相关度: {sum(r.get('score', 0) for r in results) / len(results):.3f}")
        except Exception as e:
            print(f"   ❌ 失败 - {e}")

        # 测试嵌入模型
        print("\n2. 测试嵌入模型:")
        try:
            test_text = "包豪斯设计理念"
            embedding = self.embedding_model.encode(test_text)
            print(f"   测试文本: {test_text}")
            print(f"   ✅ 成功 - 维度: {embedding.shape}")
        except Exception as e:
            print(f"   ❌ 失败 - {e}")

        # 测试作者搜索（延迟初始化）
        print("\n3. 测试作者搜索功能:")
        try:
            self.init_author_analyzer()
            test_authors = self.author_analyzer.get_author_list(limit=3)

            if test_authors:
                for author, count in test_authors:
                    print(f"   测试作者: {author}")
                    result = self.author_analyzer.analyze_author(author)
                    if result['status'] == 'found':
                        print(f"   ✅ 成功 - 找到 {result['total_publications']} 篇文章")
                    else:
                        print(f"   ❌ 失败 - {result['message']}")
                    break  # 只测试一个作者
            else:
                print("   ❌ 无作者数据可测试")
        except Exception as e:
            print(f"   ❌ 作者测试失败 - {e}")

        # 测试概念搜索（延迟初始化）
        print("\n4. 测试概念搜索功能:")
        try:
            self.init_concept_analyzer()
            test_concepts = ['包豪斯', '现代主义', '设计']

            for concept in test_concepts[:1]:  # 只测试一个概念
                print(f"   测试概念: {concept}")
                first_result = self.concept_analyzer.find_first_appearance(concept)
                if first_result['status'] == 'found':
                    print(f"   ✅ 成功 - 首次出现: {first_result['year']}")
                else:
                    print(f"   ⚠️  部分成功 - {first_result.get('message', '未知状态')}")
                break
        except Exception as e:
            print(f"   ❌ 概念测试失败 - {e}")

        print("\n🎉 测试完成！")


def main():
    """主函数"""
    try:
        system = FastRAGInteractiveSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 系统启动失败: {e}")

        # 在调试模式下显示详细错误
        if os.getenv('DEBUG'):
            import traceback
            traceback.print_exc()
        else:
            print("💡 设置环境变量 DEBUG=1 查看详细错误信息")


if __name__ == "__main__":
    main()