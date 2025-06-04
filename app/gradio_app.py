"""
gradio_app.py
艺术设计RAG系统 - Gradio界面

运行方式:
python gradio_app.py
"""

import gradio as gr
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import json
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入RAG组件
from src.rag.engine import RAGEngine
from src.analysis.concept_analyzer import ConceptAnalyzer
from src.analysis.author_analyzer import AuthorAnalyzer
from model_config import get_model_info

# 全局变量
rag_engine = None
concept_analyzer = None
author_analyzer = None
search_history = []

# 样式配置
custom_css = """
    .search-result {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric {
        text-align: center;
        padding: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        margin: 5px;
    }
"""

def initialize_system():
    """初始化RAG系统"""
    global rag_engine, concept_analyzer, author_analyzer
    
    try:
        start_time = time.time()
        
        # 初始化RAG引擎
        rag_engine = RAGEngine(
            collection_name="art_design_docs_v2",
            persist_directory="data/chroma_v2",
            enable_bm25=True
        )
        
        init_time = time.time() - start_time
        
        # 获取统计信息
        stats = rag_engine.get_statistics()
        
        return f"""
        ✅ 系统初始化成功！
        
        **初始化时间**: {init_time:.2f}秒
        **文档总数**: {stats['total_documents']:,}
        **嵌入维度**: {stats['embedding_dim']}
        **BM25状态**: {'已启用' if stats['bm25_indexed'] else '未启用'}
        **生成器状态**: {'可用' if stats['generator_available'] else '不可用'}
        """
    except Exception as e:
        return f"❌ 初始化失败: {str(e)}"

def search_documents(query, method="hybrid", top_k=5, bm25_weight=0.3, vector_weight=0.7):
    """搜索文档"""
    global rag_engine, search_history
    
    if not rag_engine:
        return "❌ 请先初始化系统", None, None
    
    if not query:
        return "❌ 请输入搜索内容", None, None
    
    try:
        start_time = time.time()
        
        # 执行搜索
        results, stats = rag_engine.hybrid_search(
            query=query,
            top_k=top_k,
            method=method,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            rerank=True
        )
        
        search_time = time.time() - start_time
        
        # 保存到历史
        search_history.append({
            'query': query,
            'results': results,
            'time': search_time,
            'timestamp': datetime.now()
        })
        
        # 格式化结果
        if results:
            output = f"🔍 搜索 '{query}' 找到 {len(results)} 个结果 (耗时 {search_time:.2f}秒)\n\n"
            
            for i, doc in enumerate(results, 1):
                metadata = doc.get('metadata', {})
                score = doc.get('rerank_score', doc.get('score', 0))
                
                output += f"""
### #{i} {metadata.get('文章名称+副标题', '无标题')}
**作者**: {metadata.get('作者名称', 'N/A')} | **年份**: {metadata.get('年份', 'N/A')} | **分类**: {metadata.get('分类', 'N/A')}
**相关度**: {score:.3f}

{doc.get('text', '')[:300]}...

---
"""
            
            # 创建结果DataFrame
            results_data = []
            for doc in results:
                metadata = doc.get('metadata', {})
                results_data.append({
                    '标题': metadata.get('文章名称+副标题', 'N/A'),
                    '作者': metadata.get('作者名称', 'N/A'),
                    '年份': metadata.get('年份', 'N/A'),
                    '分类': metadata.get('分类', 'N/A'),
                    '相关度': doc.get('score', 0)
                })
            
            df = pd.DataFrame(results_data)
            
            # 生成AI答案（如果可用）
            ai_answer = None
            if rag_engine.generator.is_available() and len(results) > 0:
                try:
                    ai_answer = rag_engine.generate_answer_with_citations(query, results[:3])
                except:
                    ai_answer = "AI答案生成失败"
            
            return output, df, ai_answer
        else:
            return "未找到相关结果", None, None
            
    except Exception as e:
        return f"❌ 搜索失败: {str(e)}", None, None

def analyze_concept(concept, analysis_type="first_appearance"):
    """分析概念"""
    global concept_analyzer, rag_engine
    
    if not rag_engine:
        return "❌ 请先初始化系统", None
    
    if not concept:
        return "❌ 请输入概念", None
    
    try:
        # 初始化概念分析器
        if not concept_analyzer:
            concept_analyzer = ConceptAnalyzer(rag_engine)
        
        if analysis_type == "first_appearance":
            result = concept_analyzer.find_first_appearance(concept)
            if result['status'] == 'found':
                output = f"""
### 概念 '{concept}' 首次出现

**年份**: {result['year']}
**文献**: {result['title']}
**作者**: {result['author']}
**分类**: {result.get('category', 'N/A')}

**上下文**:
{result['context']}

**统计**: 共找到 {result['total_docs_found']} 篇相关文献
"""
                return output, None
            else:
                return f"未找到概念 '{concept}' 的相关信息", None
                
        elif analysis_type == "timeline":
            temporal_df = concept_analyzer.analyze_temporal_distribution(concept)
            if not temporal_df.empty:
                # 创建时间线图表
                fig = px.line(
                    temporal_df, 
                    x='year', 
                    y='count',
                    title=f"'{concept}' 概念的时间分布",
                    markers=True
                )
                
                stats = f"""
### 概念 '{concept}' 时间分布统计

**总出现次数**: {temporal_df['count'].sum()}
**活跃年份数**: {len(temporal_df)}
**峰值年份**: {temporal_df.loc[temporal_df['count'].idxmax(), 'year']}
**峰值次数**: {temporal_df['count'].max()}
"""
                return stats, fig
            else:
                return "未找到时间分布数据", None
                
        elif analysis_type == "related":
            related = concept_analyzer.find_related_concepts(concept, top_n=10)
            if related:
                output = f"### 与 '{concept}' 相关的概念\n\n"
                for i, (related_concept, count) in enumerate(related, 1):
                    output += f"{i}. **{related_concept}** (共现 {count} 次)\n"
                
                # 创建条形图
                df = pd.DataFrame(related, columns=['概念', '共现次数'])
                fig = px.bar(df, x='概念', y='共现次数', title=f"与 '{concept}' 相关的概念")
                
                return output, fig
            else:
                return "未找到相关概念", None
                
    except Exception as e:
        return f"❌ 分析失败: {str(e)}", None

def analyze_author(author_name):
    """分析作者"""
    global author_analyzer, rag_engine
    
    if not rag_engine:
        return "❌ 请先初始化系统", None
    
    if not author_name:
        return "❌ 请输入作者姓名", None
    
    try:
        # 初始化作者分析器
        if not author_analyzer:
            author_analyzer = AuthorAnalyzer(rag_engine, lazy_load=True)
        
        result = author_analyzer.analyze_author(author_name)
        
        if result['status'] == 'found':
            output = f"""
### 作者分析: {result['matched_author']}

**发文总数**: {result['total_publications']}
**匹配度**: {result['match_confidence']:.2%}

#### 发表文章 (最近15篇)
"""
            for i, pub in enumerate(result['publications'][:15], 1):
                year = pub.get('year', '未知')
                output += f"{i}. [{year}] {pub['title']}\n"
            
            # 创建年份分布图
            if result['year_distribution']:
                years_df = pd.DataFrame(
                    list(result['year_distribution'].items()),
                    columns=['年份', '文章数']
                )
                fig = px.bar(years_df, x='年份', y='文章数', 
                            title=f"{result['matched_author']} 的发文时间分布")
                return output, fig
            else:
                return output, None
        else:
            return f"未找到作者 '{author_name}'", None
            
    except Exception as e:
        return f"❌ 分析失败: {str(e)}", None

def get_search_history():
    """获取搜索历史"""
    global search_history
    
    if not search_history:
        return "暂无搜索历史"
    
    output = "### 搜索历史\n\n"
    for i, item in enumerate(reversed(search_history[-10:]), 1):
        output += f"{i}. **{item['query']}** - {item['timestamp'].strftime('%H:%M:%S')} ({item['time']:.2f}秒)\n"
    
    return output

# 创建Gradio界面
with gr.Blocks(title="艺术设计RAG系统", css=custom_css) as app:
    gr.Markdown("# 艺术设计文献智能检索系统")
    
    with gr.Tab("系统初始化"):
        gr.Markdown("### 系统状态")
        init_button = gr.Button("初始化系统", variant="primary")
        init_output = gr.Markdown()
        
        init_button.click(
            fn=initialize_system,
            outputs=init_output
        )
        
        # 模型信息
        with gr.Accordion("📊 模型缓存信息", open=False):
            model_info_button = gr.Button("查看模型信息")
            model_info_output = gr.JSON()
            
            model_info_button.click(
                fn=lambda: get_model_info(),
                outputs=model_info_output
            )
    
    with gr.Tab("🔍 文献检索"):
        with gr.Row():
            with gr.Column(scale=3):
                search_input = gr.Textbox(
                    label="搜索查询",
                    placeholder="输入搜索内容，例如：包豪斯的设计理念",
                    lines=1
                )
            with gr.Column(scale=1):
                search_button = gr.Button("🔍 搜索", variant="primary")
        
        with gr.Row():
            method_select = gr.Radio(
                ["hybrid", "vector", "bm25"],
                value="hybrid",
                label="检索方法"
            )
            top_k_slider = gr.Slider(1, 20, 5, step=1, label="返回结果数")
        
        with gr.Row():
            bm25_weight = gr.Slider(0, 1, 0.3, step=0.1, label="BM25权重")
            vector_weight = gr.Slider(0, 1, 0.7, step=0.1, label="向量权重")
        
        # 搜索结果
        search_output = gr.Markdown()
        results_table = gr.DataFrame()
        
        with gr.Accordion("🤖 AI综合答案", open=False):
            ai_answer_output = gr.Markdown()
        
        search_button.click(
            fn=search_documents,
            inputs=[search_input, method_select, top_k_slider, bm25_weight, vector_weight],
            outputs=[search_output, results_table, ai_answer_output]
        )
        
        # 快速搜索按钮
        gr.Markdown("### 热门搜索")
        with gr.Row():
            gr.Button("包豪斯").click(
                lambda: ("包豪斯", "hybrid", 5, 0.3, 0.7),
                outputs=[search_input, method_select, top_k_slider, bm25_weight, vector_weight]
            )
            gr.Button("现代主义").click(
                lambda: ("现代主义", "hybrid", 5, 0.3, 0.7),
                outputs=[search_input, method_select, top_k_slider, bm25_weight, vector_weight]
            )
            gr.Button("工业设计").click(
                lambda: ("工业设计", "hybrid", 5, 0.3, 0.7),
                outputs=[search_input, method_select, top_k_slider, bm25_weight, vector_weight]
            )
    
    with gr.Tab("💡 概念分析"):
        concept_input = gr.Textbox(
            label="输入概念",
            placeholder="例如：包豪斯、现代主义、极简主义"
        )
        
        analysis_type = gr.Radio(
            ["first_appearance", "timeline", "related"],
            value="first_appearance",
            label="分析类型",
            info="选择要进行的分析"
        )
        
        analyze_button = gr.Button("分析概念", variant="primary")
        
        concept_output = gr.Markdown()
        concept_plot = gr.Plot()
        
        analyze_button.click(
            fn=analyze_concept,
            inputs=[concept_input, analysis_type],
            outputs=[concept_output, concept_plot]
        )
    
    with gr.Tab("👤 作者分析"):
        author_input = gr.Textbox(
            label="输入作者姓名",
            placeholder="例如：张三、李四"
        )
        
        author_button = gr.Button("分析作者", variant="primary")
        
        author_output = gr.Markdown()
        author_plot = gr.Plot()
        
        author_button.click(
            fn=analyze_author,
            inputs=author_input,
            outputs=[author_output, author_plot]
        )
    
    with gr.Tab("📝 搜索历史"):
        history_button = gr.Button("刷新历史")
        history_output = gr.Markdown()
        
        history_button.click(
            fn=get_search_history,
            outputs=history_output
        )

# 启动应用
if __name__ == "__main__":
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True
    )