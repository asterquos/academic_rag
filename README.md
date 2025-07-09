# 艺术设计RAG智能检索系统

基于RAG（检索增强生成）技术的艺术设计领域知识问答系统，支持879篇学术文献的智能检索与AI答案生成。

## 系统特点

- 🔍 **混合检索**：融合向量检索与BM25文本检索，兼顾语义理解和关键词匹配
- 🤖 **AI答案生成**：基于Gemini-2.0-Flash生成专业答案，包含完整引用
- 📚 **领域优化**：针对中文艺术设计文献优化，使用BGE-Large-zh嵌入模型
- 🌐 **Web界面**：简洁优雅的交互界面，支持多种检索参数配置

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/asterquos/academic_rag.git
cd academic_rag
```

### 2. 安装依赖
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 构建系统
```bash
# Step 1: 数据预处理
python -m src.preprocessing.pipeline

# Step 2: 构建向量库（需要GPU，约10-20分钟）
python build_rag.py

# Step 3: 启动Web服务
python web_app.py
```

### 4. 访问系统
打开浏览器访问：http://localhost:8000

## 系统要求

- **操作系统**：Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- **Python**：3.8 - 3.10（推荐3.10）
- **硬件要求**：
  - CPU：4核以上
  - 内存：16GB以上
  - GPU：NVIDIA GPU，16GB显存（推荐RTX 3090/4090）
  - 存储：100GB可用空间

## 可选配置

### 启用AI答案生成
```bash
# 设置Google API密钥
export GOOGLE_API_KEY="你的API密钥"  # Linux/Mac
# 或
set GOOGLE_API_KEY=你的API密钥  # Windows
```
> 注：不设置API密钥仍可使用检索功能

## 项目结构
```
academic_rag/
├── data/                # 数据目录
│   ├── raw/            # 原始Excel数据（已包含）
│   ├── processed/      # 预处理数据（自动生成）
│   └── chroma_v2/      # 向量数据库（自动生成）
├── src/                # 源代码
│   ├── rag/           # RAG核心组件
│   ├── preprocessing/ # 数据预处理
│   └── analysis/      # 分析工具
├── static/            # Web前端文件
├── web_app.py        # Web服务入口
├── build_rag.py      # 向量库构建脚本
└── requirements.txt  # 依赖列表
```

## 使用说明

1. **基础检索**：直接输入查询内容，点击搜索
2. **高级选项**：
   - 返回结果数
   - 检索方法：混合检索/向量检索/关键词检索
   - 启用AI生成：需要Google API Key

## 常见问题

**Q: ChromaDB版本兼容性错误？**  
```bash
# 删除旧文件并重建
rm -rf data/chroma_v2/
python build_rag.py
```

**Q: 向量数据库为空？**  
```bash
# 确保按顺序执行
python -m src.preprocessing.pipeline
python build_rag.py
python web_app.py
```

**Q: 端口被占用？**  
```bash
python web_app.py --port 8001
```

## 技术栈

- **后端**：FastAPI, Python 3.10
- **前端**：原生JavaScript, CSS3
- **向量数据库**：ChromaDB
- **嵌入模型**：BAAI/bge-large-zh-v1.5
- **生成模型**：Google Gemini-2.0-Flash
- **检索算法**：HNSW + BM25
