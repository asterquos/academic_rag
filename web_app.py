"""
web_app.py
艺术设计RAG系统Web应用后端
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import logging
import time
import markdown
from datetime import datetime

# 导入RAG系统组件
from src.rag.engine import RAGEngine
from src.rag.embedding import EmbeddingModel
from src.analysis.author_analyzer import AuthorAnalyzer
from src.analysis.concept_analyzer import ConceptAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="艺术设计RAG系统",
    description="基于RAG的艺术设计文献智能检索与分析系统",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求和响应模型
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    enable_generation: bool = True
    search_method: str = "hybrid"
    bm25_weight: float = 0.3
    vector_weight: float = 0.7


class SearchResult(BaseModel):
    id: str
    title: str
    author: str
    year: Optional[int]
    category: Optional[str]
    score: float
    text_preview: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    search_time: float
    total_results: int
    results: List[SearchResult]
    generated_answer: Optional[str] = None
    answer_html: Optional[str] = None
    search_stats: Dict[str, Any]


class AuthorQueryRequest(BaseModel):
    author_name: str


class ConceptQueryRequest(BaseModel):
    concept: str
    analysis_type: str = "overview"  # overview, first_appearance, evolution, related


class SystemStats(BaseModel):
    total_documents: int
    collection_name: str
    embedding_model: str
    embedding_dim: int
    bm25_enabled: bool
    generator_available: bool
    api_status: str


# 全局变量存储系统实例
rag_system = None
author_analyzer = None
concept_analyzer = None


def initialize_system():
    """初始化RAG系统"""
    global rag_system, author_analyzer, concept_analyzer

    logger.info("初始化RAG系统...")

    try:
        # 初始化嵌入模型
        embedding_model = EmbeddingModel(
            model_type='bge-large-zh',
            use_fp16=True,
            batch_size=32
        )

        # 初始化RAG引擎
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        rag_system = RAGEngine(
            collection_name='art_design_docs_v2',
            persist_directory='data/chroma_v2',
            embedding_model=embedding_model,
            gemini_api_key=api_key,
            enable_bm25=True
        )

        # 1. 使用 Elasticsearch 后端
        # rag_engine = RAGEngine(
        #     collection_name='art_design_docs',
        #     retriever_backend='elasticsearch',  # 使用 Elasticsearch
        #     es_host='localhost:9200'
        # )

        # 检查数据库状态
        doc_count = rag_system.vector_store.count()
        logger.info(f"向量数据库包含 {doc_count} 个文档")

        if doc_count == 0:
            logger.warning("向量数据库为空！请先运行 build_rag.py 构建索引")

        # 延迟初始化分析器
        author_analyzer = None
        concept_analyzer = None

        logger.info("RAG系统初始化完成")

    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化系统"""
    initialize_system()


@app.get("/")
async def root():
    """根路径，返回前端页面"""
    return FileResponse('static/index.html')


@app.get("/api/stats")
async def get_system_stats() -> SystemStats:
    """获取系统统计信息"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化")

    stats = rag_system.get_statistics()

    return SystemStats(
        total_documents=stats['total_documents'],
        collection_name=stats['collection_name'],
        embedding_model=stats['embedding_model'],
        embedding_dim=stats['embedding_dim'],
        bm25_enabled=stats['bm25_enabled'],
        generator_available=stats['generator_available'],
        api_status="正常" if stats['total_documents'] > 0 else "数据库为空"
    )


@app.post("/api/search")
async def search(request: QueryRequest) -> QueryResponse:
    """执行搜索"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化")

    start_time = time.time()

    try:
        # 执行混合搜索
        results, stats = rag_system.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            method=request.search_method,
            bm25_weight=request.bm25_weight,
            vector_weight=request.vector_weight,
            rerank=True
        )

        # 格式化搜索结果
        formatted_results = []
        for doc in results:
            metadata = doc.get('metadata', {})
            formatted_results.append(SearchResult(
                id=doc.get('id', ''),
                title=metadata.get('文章名称+副标题', '无标题'),
                author=metadata.get('作者名称', '未知作者'),
                year=metadata.get('年份'),
                category=metadata.get('分类'),
                score=doc.get('rerank_score', doc.get('score', 0)),
                text_preview=doc.get('text', '')[:300] + '...',
                metadata=metadata
            ))

        # 生成AI答案（如果启用）
        generated_answer = None
        answer_html = None

        if request.enable_generation and rag_system.generator.is_available() and results:
            try:
                # 使用前3个最相关的文档生成答案
                generated_answer = rag_system.generate_answer_with_citations(
                    request.query,
                    results[:3]
                )

                # 转换Markdown为HTML
                md = markdown.Markdown(extensions=['extra', 'codehilite'])
                answer_html = md.convert(generated_answer)

            except Exception as e:
                logger.error(f"生成答案失败: {e}")
                generated_answer = f"生成答案时出错: {str(e)}"

        # 计算搜索时间
        search_time = time.time() - start_time

        return QueryResponse(
            query=request.query,
            search_time=search_time,
            total_results=len(results),
            results=formatted_results,
            generated_answer=generated_answer,
            answer_html=answer_html,
            search_stats=stats
        )

    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/author")
async def analyze_author(request: AuthorQueryRequest):
    """分析作者"""
    global author_analyzer

    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化")

    # 延迟初始化作者分析器
    if not author_analyzer:
        logger.info("初始化作者分析器...")
        author_analyzer = AuthorAnalyzer(rag_system, lazy_load=True)

    try:
        result = author_analyzer.analyze_author(request.author_name)
        return result
    except Exception as e:
        logger.error(f"作者分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/concept")
async def analyze_concept(request: ConceptQueryRequest):
    """分析概念"""
    global concept_analyzer

    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化")

    # 延迟初始化概念分析器
    if not concept_analyzer:
        logger.info("初始化概念分析器...")
        concept_analyzer = ConceptAnalyzer(rag_system)

    try:
        if request.analysis_type == "first_appearance":
            result = concept_analyzer.find_first_appearance(request.concept)
        elif request.analysis_type == "evolution":
            result = concept_analyzer.analyze_concept_evolution(request.concept)
        elif request.analysis_type == "related":
            result = concept_analyzer.find_related_concepts(request.concept)
        else:  # overview
            result = concept_analyzer.generate_concept_report(request.concept)

        return result
    except Exception as e:
        logger.error(f"概念分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_initialized": rag_system is not None
    }


# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")


def main():
    """主函数"""
    # 创建静态文件目录
    Path("static").mkdir(exist_ok=True)

    # 启动服务器
    logger.info("启动Web服务器...")
    logger.info("访问 http://localhost:8000 使用系统")

    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()