"""
streamlit_app.py
艺术设计RAG系统 - Streamlit Web界面

运行方式:
streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime
import time
import json
from pathlib import Path
import sys

# 获取项目根目录的绝对路径
project_root = Path(__file__).resolve().parent.parent
print(f"项目根目录: {project_root}")

# 将项目根目录添加到 Python 路径
sys.path.insert(0, str(project_root))

# 导入RAG组件
from src.rag.engine import RAGEngine
from src.analysis.concept_analyzer import ConceptAnalyzer
from src.analysis.author_analyzer import AuthorAnalyzer
from model_config import get_model_info, MODEL_CACHE_DIR

# 页面配置
st.set_page_config(
    page_title="艺术设计RAG系统",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .search-result {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #ffeb3b;
        padding: 0.2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
    st.session_state.concept_analyzer = None
    st.session_state.author_analyzer = None
    st.session_state.search_history = []
    st.session_state.current_results = None
    st.session_state.init_time = None

# 侧边栏 - 系统控制
with st.sidebar:
    st.title("艺术设计RAG系统")
    st.markdown("---")
    
    # 系统状态
    st.subheader("系统状态")
    
    if st.session_state.rag_engine is None:
        if st.button("初始化系统", type="primary", use_container_width=True):
            with st.spinner("正在初始化系统..."):
                start_time = time.time()
                try:
                    # 初始化RAG引擎
                    st.session_state.rag_engine = RAGEngine(
                        collection_name="art_design_docs_v2",
                        persist_directory="data/chroma_v2",
                        enable_bm25=True
                    )
                    st.session_state.init_time = time.time() - start_time
                    st.success(f"✅ 系统初始化成功 ({st.session_state.init_time:.2f}秒)")
                except Exception as e:
                    st.error(f"❌ 初始化失败: {str(e)}")
    else:
        st.success("✅ 系统已就绪")
        
        # 显示系统统计
        stats = st.session_state.rag_engine.get_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("文档总数", f"{stats['total_documents']:,}")
            st.metric("嵌入维度", stats['embedding_dim'])
        with col2:
            st.metric("BM25索引", "✅" if stats['bm25_indexed'] else "❌")
            st.metric("生成器", "✅" if stats['generator_available'] else "❌")
        
        # 模型信息
        with st.expander("模型信息"):
            model_info = get_model_info()
            st.write(f"**缓存目录**: `{model_info['cache_directory']}`")
            if model_info['cached_models']:
                st.write("**已缓存模型**:")
                for model in model_info['cached_models']:
                    st.write(f"- {model['name']}: {model['size_mb']:.1f} MB")
    
    st.markdown("---")
    
    # 搜索设置
    st.subheader("⚙️ 搜索设置")
    
    search_method = st.selectbox(
        "检索方法",
        ["hybrid", "vector", "bm25"],
        help="hybrid: 混合检索（推荐）\nvector: 纯向量检索\nbm25: 纯关键词检索"
    )
    
    top_k = st.slider("返回结果数", 1, 20, 5)
    
    if search_method == "hybrid":
        col1, col2 = st.columns(2)
        with col1:
            bm25_weight = st.slider("BM25权重", 0.0, 1.0, 0.3)
        with col2:
            vector_weight = st.slider("向量权重", 0.0, 1.0, 0.7)
    else:
        bm25_weight = 0.3
        vector_weight = 0.7
    
    enable_rerank = st.checkbox("启用重排序", value=True)
    
    # 清除历史
    if st.button("🗑️ 清除搜索历史"):
        st.session_state.search_history = []
        st.session_state.current_results = None
        st.success("搜索历史已清除")

# 主界面
st.markdown('<h1 class="main-header">🎨 艺术设计文献智能检索系统</h1>', unsafe_allow_html=True)

# 检查系统是否初始化
if st.session_state.rag_engine is None:
    st.info("👈 请先在侧边栏初始化系统")
    st.stop()

# 创建标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 文献检索", "💡 概念分析", "👤 作者分析", "📊 数据可视化", "📝 搜索历史"])

# Tab 1: 文献检索
with tab1:
    st.subheader("智能文献检索")
    
    # 搜索框
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "输入搜索查询",
            placeholder="例如：包豪斯的设计理念、现代主义建筑、工业设计发展...",
            key="search_query"
        )
    with col2:
        search_button = st.button("🔍 搜索", type="primary", use_container_width=True)
    
    # 快速搜索建议
    st.markdown("**热门搜索**:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("包豪斯"):
            query = "包豪斯"
    with col2:
        if st.button("现代主义"):
            query = "现代主义"
    with col3:
        if st.button("工业设计"):
            query = "工业设计"
    with col4:
        if st.button("装饰艺术"):
            query = "装饰艺术"
    
    # 执行搜索
    if search_button and query:
        with st.spinner(f"正在搜索 '{query}'..."):
            start_time = time.time()
            
            try:
                # 执行搜索
                results, stats = st.session_state.rag_engine.hybrid_search(
                    query=query,
                    top_k=top_k,
                    method=search_method,
                    bm25_weight=bm25_weight,
                    vector_weight=vector_weight,
                    rerank=enable_rerank
                )
                
                search_time = time.time() - start_time
                
                # 保存结果
                st.session_state.current_results = {
                    'query': query,
                    'results': results,
                    'stats': stats,
                    'search_time': search_time,
                    'timestamp': datetime.now()
                }
                
                # 添加到历史
                st.session_state.search_history.append(st.session_state.current_results)
                
            except Exception as e:
                st.error(f"搜索失败: {str(e)}")
                results = []
    
    # 显示搜索结果
    if st.session_state.current_results:
        results = st.session_state.current_results['results']
        search_time = st.session_state.current_results['search_time']
        query = st.session_state.current_results['query']
        
        # 搜索统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("搜索耗时", f"{search_time:.2f}秒")
        with col2:
            st.metric("结果数量", len(results))
        with col3:
            st.metric("检索方法", search_method.upper())
        with col4:
            if results:
                avg_score = sum(r.get('score', 0) for r in results) / len(results)
                st.metric("平均相关度", f"{avg_score:.3f}")
        
        st.markdown("---")
        
        # 显示结果
        if results:
            st.subheader(f"搜索结果：{query}")
            
            for i, doc in enumerate(results, 1):
                metadata = doc.get('metadata', {})
                score = doc.get('rerank_score', doc.get('score', 0))
                
                with st.container():
                    st.markdown(f"""
                    <div class="search-result">
                        <h4>#{i} {metadata.get('文章名称+副标题', '无标题')}</h4>
                        <p><strong>作者:</strong> {metadata.get('作者名称', 'N/A')} | 
                           <strong>年份:</strong> {metadata.get('年份', 'N/A')} | 
                           <strong>分类:</strong> {metadata.get('分类', 'N/A')} |
                           <strong>相关度:</strong> {score:.3f}</p>
                        <p>{doc.get('text', '')[:500]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 展开查看完整内容
                    with st.expander("查看完整内容"):
                        st.write(doc.get('text', ''))
                        st.write("**元数据**:")
                        st.json(metadata)
            
            # AI生成答案（如果可用）
            if st.session_state.rag_engine.generator.is_available():
                st.markdown("---")
                if st.button("🤖 生成AI综合答案"):
                    with st.spinner("AI正在生成答案..."):
                        try:
                            answer = st.session_state.rag_engine.generate_answer_with_citations(
                                query, results[:3]
                            )
                            st.markdown("### 🤖 AI综合答案")
                            st.markdown(answer)
                        except Exception as e:
                            st.error(f"生成答案失败: {str(e)}")
        else:
            st.info("未找到相关结果")

# Tab 2: 概念分析
with tab2:
    st.subheader("概念演进分析")
    
    # 初始化概念分析器
    if st.session_state.concept_analyzer is None and st.button("初始化概念分析器"):
        with st.spinner("正在初始化..."):
            st.session_state.concept_analyzer = ConceptAnalyzer(st.session_state.rag_engine)
            st.success("概念分析器就绪")
    
    if st.session_state.concept_analyzer:
        # 概念输入
        concept = st.text_input("输入要分析的概念", placeholder="例如：包豪斯、现代主义、极简主义...")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            analyze_first = st.button("🔍 首次出现")
        with col2:
            analyze_timeline = st.button("📈 时间分布")
        with col3:
            analyze_related = st.button("🔗 相关概念")
        with col4:
            analyze_evolution = st.button("📊 演进分析")
        
        # 首次出现分析
        if analyze_first and concept:
            with st.spinner(f"查找 '{concept}' 首次出现..."):
                result = st.session_state.concept_analyzer.find_first_appearance(concept)
                
                if result['status'] == 'found':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("首次出现年份", result['year'])
                    with col2:
                        st.metric("文献数量", result['total_docs_found'])
                    with col3:
                        st.metric("有效年份文档", result['docs_with_valid_year'])
                    
                    st.write(f"**文献标题**: {result['title']}")
                    st.write(f"**作者**: {result['author']}")
                    st.write(f"**分类**: {result.get('category', 'N/A')}")
                    
                    with st.expander("查看上下文"):
                        st.text(result['context'])
                else:
                    st.warning(result.get('message', '未找到相关信息'))
        
        # 时间分布分析
        if analyze_timeline and concept:
            with st.spinner(f"分析 '{concept}' 时间分布..."):
                temporal_df = st.session_state.concept_analyzer.analyze_temporal_distribution(concept)
                
                if not temporal_df.empty:
                    # 创建时间线图表
                    fig = px.line(
                        temporal_df, 
                        x='year', 
                        y='count',
                        title=f"'{concept}' 概念的时间分布",
                        labels={'year': '年份', 'count': '出现次数'},
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示统计
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总出现次数", temporal_df['count'].sum())
                    with col2:
                        st.metric("活跃年份数", len(temporal_df))
                    with col3:
                        peak_year = temporal_df.loc[temporal_df['count'].idxmax(), 'year']
                        st.metric("峰值年份", peak_year)
                    with col4:
                        st.metric("峰值次数", temporal_df['count'].max())
                    
                    # 显示详细数据
                    with st.expander("查看详细数据"):
                        st.dataframe(temporal_df)
                else:
                    st.info("未找到时间分布数据")
        
        # 相关概念分析
        if analyze_related and concept:
            with st.spinner(f"查找与 '{concept}' 相关的概念..."):
                related = st.session_state.concept_analyzer.find_related_concepts(concept, top_n=15)
                
                if related:
                    # 创建词云效果的条形图
                    related_df = pd.DataFrame(related, columns=['概念', '共现次数'])
                    
                    fig = px.bar(
                        related_df,
                        x='共现次数',
                        y='概念',
                        orientation='h',
                        title=f"与 '{concept}' 相关的概念",
                        color='共现次数',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("未找到相关概念")
        
        # 演进分析
        if analyze_evolution and concept:
            with st.spinner(f"分析 '{concept}' 的演进历程..."):
                evolution = st.session_state.concept_analyzer.analyze_concept_evolution(concept)
                
                if evolution['status'] == 'success':
                    overview = evolution['overview']
                    
                    # 概览卡片
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("时间跨度", f"{overview['span_years']}年")
                    with col2:
                        st.metric("总提及数", overview['total_mentions'])
                    with col3:
                        st.metric("整体趋势", overview['trend'])
                    with col4:
                        st.metric("峰值年份", overview['peak_year'])
                    
                    # 分期分析
                    if evolution.get('periods'):
                        st.subheader("分期演进分析")
                        periods_df = pd.DataFrame(evolution['periods'])
                        
                        fig = px.bar(
                            periods_df,
                            x='period',
                            y='total_mentions',
                            title="各时期提及次数",
                            labels={'period': '时期', 'total_mentions': '提及次数'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(evolution.get('message', '分析失败'))
    else:
        st.info("请先初始化概念分析器")

# Tab 3: 作者分析
with tab3:
    st.subheader("作者研究分析")
    
    # 初始化作者分析器
    if st.session_state.author_analyzer is None and st.button("初始化作者分析器"):
        with st.spinner("正在初始化（可能需要几秒钟）..."):
            st.session_state.author_analyzer = AuthorAnalyzer(
                st.session_state.rag_engine, 
                lazy_load=True
            )
            st.success("作者分析器就绪")
    
    if st.session_state.author_analyzer:
        # 作者搜索
        col1, col2 = st.columns([3, 1])
        with col1:
            author_query = st.text_input("输入作者姓名", placeholder="例如：张三、李四...")
        with col2:
            search_author = st.button("🔍 分析作者", type="primary")
        
        # 显示热门作者
        if st.button("📊 显示热门作者"):
            with st.spinner("加载作者列表..."):
                author_list = st.session_state.author_analyzer.get_author_list(limit=20)
                
                if author_list:
                    st.subheader("发文量前20的作者")
                    
                    # 创建条形图
                    authors_df = pd.DataFrame(author_list, columns=['作者', '文章数'])
                    fig = px.bar(
                        authors_df,
                        x='文章数',
                        y='作者',
                        orientation='h',
                        title="作者发文量排行",
                        color='文章数',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
        
        # 作者分析
        if search_author and author_query:
            with st.spinner(f"分析作者 '{author_query}'..."):
                result = st.session_state.author_analyzer.analyze_author(author_query)
                
                if result['status'] == 'found':
                    # 基本信息
                    st.success(f"找到作者: {result['matched_author']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("发文总数", result['total_publications'])
                    with col2:
                        st.metric("匹配度", f"{result['match_confidence']:.2%}")
                    with col3:
                        if result['year_distribution']:
                            year_range = f"{min(result['year_distribution'].keys())}-{max(result['year_distribution'].keys())}"
                            st.metric("活跃年份", year_range)
                    
                    # 年份分布图
                    if result['year_distribution']:
                        st.subheader("发文年份分布")
                        years_df = pd.DataFrame(
                            list(result['year_distribution'].items()),
                            columns=['年份', '文章数']
                        )
                        fig = px.bar(
                            years_df,
                            x='年份',
                            y='文章数',
                            title=f"{result['matched_author']} 的发文时间分布"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 研究领域
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result['category_distribution']:
                            st.subheader("研究分类")
                            categories_df = pd.DataFrame(
                                list(result['category_distribution'].items())[:10],
                                columns=['分类', '文章数']
                            )
                            fig = px.pie(
                                categories_df,
                                values='文章数',
                                names='分类',
                                title="研究分类分布"
                            )
                            st.plotly_chart(fig)
                    
                    with col2:
                        if result['research_topics']:
                            st.subheader("研究主题")
                            topics_df = pd.DataFrame(
                                list(result['research_topics'].items()),
                                columns=['主题', '出现次数']
                            )
                            fig = px.bar(
                                topics_df,
                                x='出现次数',
                                y='主题',
                                orientation='h',
                                title="主要研究主题"
                            )
                            st.plotly_chart(fig)
                    
                    # 合作者网络
                    if result['collaborators']:
                        st.subheader("主要合作者")
                        collaborators_df = pd.DataFrame(
                            list(result['collaborators'].items()),
                            columns=['合作者', '合作次数']
                        )
                        st.dataframe(collaborators_df)
                    
                    # 发表文章列表
                    with st.expander("查看发表文章"):
                        publications_df = pd.DataFrame(result['publications'])
                        st.dataframe(
                            publications_df[['year', 'title', 'category']].sort_values('year', ascending=False),
                            use_container_width=True
                        )
                else:
                    st.warning(result.get('message', '未找到作者'))
                    
                    # 提供相似建议
                    similar = st.session_state.author_analyzer.search_authors(author_query)
                    if similar:
                        st.info("您可能要找的作者:")
                        for author, count in similar[:5]:
                            if st.button(f"{author} ({count}篇)", key=f"suggest_{author}"):
                                author_query = author
    else:
        st.info("请先初始化作者分析器")

# Tab 4: 数据可视化
with tab4:
    st.subheader("数据可视化分析")
    
    # 选择可视化类型
    viz_type = st.selectbox(
        "选择可视化类型",
        ["年份分布统计", "分类分布分析", "概念关系网络", "作者合作网络"]
    )
    
    if viz_type == "年份分布统计":
        if st.button("生成年份分布图"):
            # 这里需要从数据库获取统计数据
            # 示例数据
            st.info("正在开发中...")
    
    elif viz_type == "分类分布分析":
        if st.button("生成分类分布图"):
            st.info("正在开发中...")
    
    elif viz_type == "概念关系网络":
        st.write("输入多个概念（用逗号分隔）:")
        concepts_input = st.text_input("概念列表", placeholder="包豪斯,现代主义,极简主义")
        
        if st.button("生成概念网络") and concepts_input:
            concepts = [c.strip() for c in concepts_input.split(',')]
            
            with st.spinner("正在生成概念关系网络..."):
                # 创建网络图
                G = nx.Graph()
                
                # 添加概念节点
                for concept in concepts:
                    G.add_node(concept, size=20)
                
                # 获取概念间的关系（示例）
                # 实际应该通过分析共现关系
                import random
                for i in range(len(concepts)):
                    for j in range(i+1, len(concepts)):
                        if random.random() > 0.5:
                            G.add_edge(concepts[i], concepts[j], weight=random.random())
                
                # 使用Plotly绘制
                pos = nx.spring_layout(G)
                
                edge_trace = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace.append(go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color='#888'),
                        hoverinfo='none'
                    ))
                
                node_trace = go.Scatter(
                    x=[pos[node][0] for node in G.nodes()],
                    y=[pos[node][1] for node in G.nodes()],
                    mode='markers+text',
                    text=[node for node in G.nodes()],
                    textposition="top center",
                    marker=dict(
                        size=30,
                        color='#1f77b4'
                    )
                )
                
                fig = go.Figure(data=edge_trace + [node_trace])
                fig.update_layout(
                    title="概念关系网络",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "作者合作网络":
        st.info("正在开发中...")

# Tab 5: 搜索历史
with tab5:
    st.subheader("搜索历史记录")
    
    if st.session_state.search_history:
        # 显示历史统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总搜索次数", len(st.session_state.search_history))
        with col2:
            avg_time = sum(h['search_time'] for h in st.session_state.search_history) / len(st.session_state.search_history)
            st.metric("平均搜索时间", f"{avg_time:.2f}秒")
        with col3:
            avg_results = sum(len(h['results']) for h in st.session_state.search_history) / len(st.session_state.search_history)
            st.metric("平均结果数", f"{avg_results:.1f}")
        
        # 显示历史记录
        for i, history in enumerate(reversed(st.session_state.search_history[-10:]), 1):
            with st.expander(f"{i}. {history['query']} - {history['timestamp'].strftime('%H:%M:%S')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**搜索方法**: {history['stats']['method']}")
                with col2:
                    st.write(f"**结果数量**: {len(history['results'])}")
                with col3:
                    st.write(f"**搜索时间**: {history['search_time']:.2f}秒")
                
                if st.button(f"重新加载结果", key=f"reload_{i}"):
                    st.session_state.current_results = history
                    st.success("结果已加载到搜索页面")
    else:
        st.info("暂无搜索历史")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>艺术设计RAG系统 v1.0 | 
        基于 Streamlit + ChromaDB + Gemini | 
        © 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)