"""
src/rag/generator.py
Enhanced LLM Answer Generator with Advanced Prompt Engineering
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型枚举"""
    DEFINITION = "definition"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    ANALYTICAL = "analytical"
    FACTUAL = "factual"
    EXPLORATORY = "exploratory"
    GENERAL = "general"


@dataclass
class GenerationConfig:
    """生成配置"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_output_tokens: int = 2048
    stop_sequences: Optional[List[str]] = None


class PromptTemplateManager:
    """提示词模板管理器"""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, str]:
        """初始化专业的提示词模板"""
        return {
            QueryType.GENERAL: """你是一位资深的艺术设计领域专家，拥有深厚的理论功底和实践经验。

核心能力：
• 精通设计史（包豪斯、现代主义、后现代主义、当代设计等）
• 熟悉中西方艺术设计理论体系
• 了解设计教育、设计批评和设计研究方法
• 掌握跨学科知识（美学、社会学、技术等）

回答原则：
1. **准确性第一**：严格基于提供的文档，不编造信息
2. **学术规范**：
   - 使用准确的专业术语
   - 保持逻辑严谨和论述完整
   - 区分事实陈述与观点评论
3. **引用规范**：
   - 直接引用：使用"根据[作者, 年份]的观点..."
   - 间接引用：使用"文献显示..."或"研究表明..."
   - 多源综合：按时间顺序或重要性排列
4. **结构清晰**：
   - 开篇概述核心观点
   - 中间展开论述，层次分明
   - 结尾总结要点
5. **诚实表达**：
   - 信息充分时给出全面回答
   - 信息不足时明确说明局限性
   - 存在争议时呈现不同观点""",

            QueryType.DEFINITION: """你是艺术设计百科全书的资深编纂者，擅长准确定义概念。

定义类问题的回答框架：

【定义】（1-2句精炼概括）
- 给出核心定义，突出本质特征

【核心特征】（3-5个要点）
- 列举区别性特征
- 说明关键要素

【历史溯源】（如文档中有相关信息）
- 概念起源
- 发展脉络
- 重要转折

【代表人物/作品】（如有）
- 主要贡献者
- 典型案例

【当代意义】
- 现实影响
- 理论价值

引用要求：
- 定义来源必须标注：[作者, 年份, 页码]
- 多个定义需对比分析异同
- 优先引用权威来源""",

            QueryType.COMPARISON: """你是专精于比较研究的设计理论学者。

比较分析框架：

【比较概述】
简要介绍比较对象及比较的意义

【共同特征】
► 历史背景的相似性
► 核心理念的共通处
► 影响范围的交集
► 方法论的相近点

【差异分析】
► 本质区别
  - 理论基础差异
  - 价值取向不同
► 表现形式
  - 风格特征对比
  - 实践方式差异
► 发展路径
  - 演进方向分化
  - 影响因素差别

【对比总结】
- 关键差异点梳理
- 各自的适用情境
- 互补或竞争关系

注意事项：
✓ 使用平行结构便于对照
✓ 避免价值优劣判断
✓ 以具体例证支撑观点
✓ 关注文化语境差异""",

            QueryType.TEMPORAL: """你是设计史研究专家，善于梳理时间脉络。

时间类问题回答模式：

【时间定位】
◆ 精确时间点：具体年份/年代
◆ 时期划分：早期/中期/晚期/转型期

【编年脉络】
┌─ 起始阶段：背景与动因
├─ 发展阶段：关键事件与转折
├─ 成熟阶段：特征与影响
└─ 演变趋势：后续发展

【时代背景】
• 社会文化环境
• 技术条件变化
• 思想观念演进

【因果关系】
→ 前因：导致产生的条件
→ 过程：发展演变的逻辑
→ 后果：产生的影响效应

时间表述规范：
- 具体年份：1919年
- 时间段：1920年代
- 相对时期：20世纪初期
- 标注信息时效性""",

            QueryType.ANALYTICAL: """你是深度思考的设计批评家和理论家。

分析性问题处理框架：

【问题解析】
明确分析对象、范围和核心议题

【多维分析】
➤ 历史维度
  - 时代背景分析
  - 历史地位评估
➤ 理论维度
  - 概念内涵剖析
  - 理论贡献评价
➤ 实践维度
  - 应用案例分析
  - 实际效果评估
➤ 文化维度
  - 文化语境考察
  - 跨文化比较

【证据链条】
- 关键论据提炼
- 文献交叉验证
- 逻辑推理过程

【批判性思考】
※ 主流观点梳理
※ 不同立场对比
※ 潜在问题识别
※ 发展可能探讨

【综合结论】
总结核心观点，提出平衡、深刻的见解""",

            QueryType.FACTUAL: """你是注重事实准确性的艺术设计文献专家。

事实类问题回答要求：

1. 直接回答核心事实
2. 提供具体数据支撑
3. 标注信息来源
4. 补充相关背景（如需要）

格式示例：
问：[事实性问题]
答：根据[来源]，[直接事实陈述]。具体而言，[详细说明]。

注意：
- 区分确定事实与推测
- 数据必须准确无误
- 日期、人名、地点等需精确""",

            QueryType.EXPLORATORY: """你是富有洞察力的设计研究者。

探索性问题引导框架：

【问题理解】
- 识别探索方向
- 明确知识边界

【现有认知】
基于文档梳理已知信息

【探索路径】
► 可能的研究方向
► 相关理论视角
► 潜在关联领域

【开放性思考】
- 提出假设性观点
- 指出研究空白
- 建议深入方向

【学术诚信】
明确标注：
- 哪些是文档支持的
- 哪些是推理延伸的
- 哪些需要进一步验证"""
        }

    def get_template(self, query_type: QueryType) -> str:
        """获取对应类型的模板"""
        return self.templates.get(query_type, self.templates[QueryType.GENERAL])


class AnswerGenerator:
    """增强的答案生成器"""

    def __init__(self,
                 api_key: str = None,
                 model_name: str = "gemini-2.0-flash",
                 enable_cot: bool = True):
        """
        初始化生成器

        Args:
            api_key: API密钥
            model_name: 模型名称
            enable_cot: 是否启用思维链
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name
        self.enable_cot = enable_cot
        self.llm = None

        # 初始化组件
        self.template_manager = PromptTemplateManager()
        self.query_analyzer = QueryAnalyzer()

        if self.api_key:
            self._init_llm()
        else:
            logger.warning("未配置API密钥，生成功能不可用")

    def _init_llm(self):
        """初始化LLM"""
        try:
            genai.configure(api_key=self.api_key)
            self.llm = genai.GenerativeModel(self.model_name)
            logger.info(f"LLM初始化成功: {self.model_name}")
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            self.llm = None

    def generate(
            self,
            query: str,
            context: List[Dict[str, Any]],
            system_prompt: Optional[str] = None,
            query_type: Optional[QueryType] = None,
            config: Optional[GenerationConfig] = None
    ) -> str:
        """
        生成答案的主方法

        Args:
            query: 用户查询
            context: 检索到的文档上下文
            system_prompt: 系统提示词（可选）
            query_type: 查询类型（可选）
            config: 生成配置（可选）

        Returns:
            生成的答案
        """
        if not self.llm:
            return "生成功能不可用（请设置GOOGLE_API_KEY或GEMINI_API_KEY环境变量）"

        # 1. 分析查询类型
        if query_type is None:
            query_type = self.query_analyzer.analyze(query)
            logger.info(f"识别的查询类型: {query_type.value}")

        # 2. 选择或使用提供的系统提示词
        if system_prompt is None:
            system_prompt = self.template_manager.get_template(query_type)

        # 3. 智能处理上下文
        processed_context = self._process_context(context, query, query_type)

        # 4. 构建完整提示词
        full_prompt = self._build_prompt(
            system_prompt=system_prompt,
            context=processed_context,
            query=query,
            query_type=query_type
        )

        # 5. 获取生成配置
        if config is None:
            config = self._get_optimal_config(query_type)

        # 6. 生成答案
        try:
            response = self.llm.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_output_tokens=config.max_output_tokens,
                    stop_sequences=config.stop_sequences
                )
            )

            # 7. 后处理
            answer = self._post_process(response.text, context, query_type)
            return answer

        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return self._generate_fallback(query, context, query_type)

    # 在 generator.py 中修改 generate_with_citations 方法

    def generate_with_citations(self, query: str, context: List[Dict[str, Any]]) -> str:
        """生成带引用的答案"""
        citation_prompt = """你是一个艺术设计领域的专家助手。请基于提供的文档信息，给出准确且带有规范引用的回答。

    引用规范要求：

    1. **完整引用格式**（首次引用某文献时）：
       格式：[作者, 年份, 《文章标题》]
       示例：[王明, 2020, 《包豪斯设计理念研究》]认为...

    2. **简化引用格式**（同一文献再次引用时）：
       格式：[作者, 年份]
       示例：正如前文提到的[王明, 2020]所述...

    3. **直接引用原文**：
       格式：根据[作者, 年份, 《文章标题》]："引用的原文内容"
       示例：根据[张三, 2019, 《现代设计史》]："包豪斯是现代设计教育的先驱"

    4. **多文献综合引用**：
       - 观点一致时：多位学者[作者1, 年份1, 《标题1》; 作者2, 年份2, 《标题2》]共同指出...
       - 观点对比时：关于X问题，[作者1, 年份1, 《标题1》]认为...，而[作者2, 年份2, 《标题2》]则提出...

    5. **特殊情况处理**：
       - 无作者信息：[佚名, 年份, 《文章标题》]
       - 无年份信息：[作者, 出版年不详, 《文章标题》]
       - 都没有时：有文献《文章标题》指出...

    回答要求：
    - 每个关键观点都需要标注完整来源（至少在首次引用时包含文章标题）
    - 保持引用格式的一致性和准确性
    - 区分原文引用和概括总结
    - 如果文档信息不足，明确指出局限性"""

        return self.generate(query, context, system_prompt=citation_prompt)

    def _process_context(self,
                         context: List[Dict[str, Any]],
                         query: str,
                         query_type: QueryType) -> str:
        """智能处理上下文"""

        # 根据查询类型确定上下文策略
        if query_type == QueryType.COMPARISON:
            return self._process_comparison_context(context, query)
        elif query_type == QueryType.TEMPORAL:
            return self._process_temporal_context(context)
        else:
            return self._process_general_context(context, query)

    def _process_general_context(self, context: List[Dict], query: str) -> str:
        """处理一般上下文"""
        # 按相关性排序
        sorted_context = sorted(
            context,
            key=lambda x: x.get('score', x.get('rerank_score', 0)),
            reverse=True
        )

        formatted_docs = []
        total_chars = 0
        max_chars = 10000  # 增加上下文容量

        # 提取查询关键词用于高亮
        query_terms = set(self._extract_keywords(query))

        for i, doc in enumerate(sorted_context, 1):
            metadata = doc.get('metadata', {})

            # 构建文档头部信息
            doc_header = self._format_doc_header(i, metadata, doc)

            # 智能提取相关内容
            text = doc.get('text', '')
            relevant_text = self._extract_relevant_content(
                text, query_terms,
                max_length=2000 if i <= 3 else 1000
            )

            # 组装文档
            doc_text = f"{doc_header}\n{relevant_text}\n{'=' * 60}\n"

            # 检查长度限制
            if total_chars + len(doc_text) > max_chars:
                if i <= 3:  # 确保至少包含前3个最相关的文档
                    remaining = max_chars - total_chars
                    doc_text = doc_text[:remaining] + "\n[内容截断...]\n"
                    formatted_docs.append(doc_text)
                break

            formatted_docs.append(doc_text)
            total_chars += len(doc_text)

        return "\n".join(formatted_docs)

    def _process_comparison_context(self, context: List[Dict], query: str) -> str:
        """处理比较类查询的上下文"""
        # 识别比较对象
        comparison_terms = self._extract_comparison_terms(query)

        # 按比较对象分组文档
        grouped_docs = {term: [] for term in comparison_terms}
        other_docs = []

        for doc in context:
            text = doc.get('text', '').lower()
            assigned = False

            for term in comparison_terms:
                if term.lower() in text:
                    grouped_docs[term].append(doc)
                    assigned = True
                    break

            if not assigned:
                other_docs.append(doc)

        # 格式化分组后的文档
        formatted_parts = []

        for term, docs in grouped_docs.items():
            if docs:
                formatted_parts.append(f"\n【关于 {term} 的文献】\n")
                for i, doc in enumerate(docs[:3], 1):
                    formatted_parts.append(self._format_single_doc(doc, i))

        if other_docs:
            formatted_parts.append("\n【综合性文献】\n")
            for i, doc in enumerate(other_docs[:2], 1):
                formatted_parts.append(self._format_single_doc(doc, i))

        return "".join(formatted_parts)

    def _process_temporal_context(self, context: List[Dict]) -> str:
        """处理时间类查询的上下文"""
        # 按时间排序
        sorted_context = sorted(
            context,
            key=lambda x: x.get('metadata', {}).get('年份', 9999)
        )

        formatted_docs = []
        current_decade = None

        for doc in sorted_context:
            year = doc.get('metadata', {}).get('年份')

            # 按年代分组
            if year:
                decade = (year // 10) * 10
                if decade != current_decade:
                    formatted_docs.append(f"\n【{decade}年代】\n")
                    current_decade = decade

            formatted_docs.append(self._format_single_doc(doc, len(formatted_docs) + 1))

        return "".join(formatted_docs)

    def _format_doc_header(self, index: int, metadata: Dict, doc: Dict) -> str:
        """格式化文档头部 - 安全版本"""
        # 安全提取并转换为字符串
        year = str(metadata.get('年份', '未知'))
        author = str(metadata.get('作者名称', '未知作者'))
        title = str(metadata.get('文章名称+副标题', '无标题'))
        category = str(metadata.get('分类', ''))
        score = doc.get('rerank_score', doc.get('score', 'N/A'))

        # 转义特殊字符
        for char in ['{', '}', '[', ']']:
            title = title.replace(char, '\\' + char)
            author = author.replace(char, '\\' + char)

        # 使用字符串拼接而不是f-string
        header = "【文档 " + str(index) + "】\n"
        header += "📄 标题：《" + title + "》\n"
        header += "👤 作者：" + author + "\n"
        header += "📅 年份：" + year + "\n"
        header += "🏷️ 分类：" + category + "\n"

        if isinstance(score, (int, float)):
            header += "📊 相关度：" + f"{score:.3f}" + "\n"
        else:
            header += "📊 相关度：" + str(score) + "\n"

        header += "----------------------------------------"

        return header

    def _format_single_doc(self, doc: Dict, index: int) -> str:
        """格式化单个文档"""
        metadata = doc.get('metadata', {})
        header = self._format_doc_header(index, metadata, doc)
        text = doc.get('text', '')[:1500]

        return f"{header}\n{text}...\n{'=' * 60}\n"

    def _extract_relevant_content(self, text: str, keywords: set, max_length: int) -> str:
        """提取相关内容"""
        if not text:
            return ""

        # 分段
        paragraphs = text.split('\n\n')

        # 计算每段的相关性分数
        scored_paragraphs = []
        for para in paragraphs:
            if len(para.strip()) < 20:
                continue

            para_lower = para.lower()
            # 计算关键词覆盖率
            keyword_score = sum(1 for kw in keywords if kw.lower() in para_lower)
            # 考虑段落位置（开头的段落通常更重要）
            position_score = 1.0 / (paragraphs.index(para) + 1)
            # 综合分数
            total_score = keyword_score + position_score * 0.3

            scored_paragraphs.append((total_score, para))

        # 按分数排序
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

        # 组合高分段落
        result_parts = []
        current_length = 0

        # 始终包含开头
        if paragraphs and len(paragraphs[0]) > 20:
            result_parts.append(paragraphs[0])
            current_length += len(paragraphs[0])

        # 添加高分段落
        for score, para in scored_paragraphs:
            if para not in result_parts and current_length + len(para) <= max_length:
                result_parts.append(para)
                current_length += len(para)

        # 如果内容太少，直接返回原文截断
        if current_length < 200:
            return text[:max_length] + "..."

        return "\n\n".join(result_parts)

    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询关键词"""
        # 简单实现：分词并过滤
        stopwords = {'的', '是', '在', '和', '了', '有', '与', '为', '等', '及', '或', '但', '而'}
        words = query.split()
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        return keywords

    def _extract_comparison_terms(self, query: str) -> List[str]:
        """提取比较对象"""
        # 识别 "A和B" "A与B" "A vs B" 等模式
        patterns = [
            r'(.+?)[和与跟同](.+?)(?:的|之间|相比|对比|区别|差异|不同)',
            r'(.+?)\s+vs\.?\s+(.+)',
            r'比较(.+?)和(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

        return []

    def _build_prompt(self,
                      system_prompt: str,
                      context: str,
                      query: str,
                      query_type: QueryType) -> str:
        """构建完整的提示词"""

        # 添加思维链引导
        cot_prompt = ""
        if self.enable_cot:
            cot_prompt = self._get_cot_prompt(query_type)

        # 构建完整提示
        full_prompt = f"""{system_prompt}

{cot_prompt}

---
📚 参考文献资料：
{context}
---

❓ 用户问题：{query}

💡 请基于以上文献资料，按照指定的框架和要求，给出专业、准确、有深度的回答：
"""

        return full_prompt

    def _get_cot_prompt(self, query_type: QueryType) -> str:
        """获取思维链提示"""
        cot_prompts = {
            QueryType.ANALYTICAL: """
分析步骤：
1. 先识别问题的核心概念和分析维度
2. 评估每篇文献与问题的相关性
3. 提取关键观点和证据
4. 构建逻辑论证结构
5. 形成综合性结论
""",
            QueryType.COMPARISON: """
比较步骤：
1. 明确比较对象和比较维度
2. 从文献中分别提取两个对象的信息
3. 识别共同点和差异点
4. 分析差异的深层原因
5. 总结比较结论
""",
            QueryType.TEMPORAL: """
时间分析步骤：
1. 建立时间坐标轴
2. 标注关键时间节点
3. 识别发展阶段和转折点
4. 分析变化的原因和影响
5. 总结演变规律
"""
        }

        return cot_prompts.get(query_type, """
思考步骤：
1. 理解问题的核心诉求
2. 评估文献的相关性和可靠性
3. 提取关键信息和观点
4. 组织论述结构
5. 得出结论并检验
""")

    def _get_optimal_config(self, query_type: QueryType) -> GenerationConfig:
        """根据查询类型获取最优生成配置"""
        configs = {
            QueryType.FACTUAL: GenerationConfig(
                temperature=0.3,  # 降低随机性
                top_p=0.8,
                top_k=20
            ),
            QueryType.ANALYTICAL: GenerationConfig(
                temperature=0.7,  # 平衡创造性和准确性
                top_p=0.9,
                top_k=40,
                max_output_tokens=3000  # 允许更长的分析
            ),
            QueryType.EXPLORATORY: GenerationConfig(
                temperature=0.8,  # 提高创造性
                top_p=0.95,
                top_k=50
            )
        }

        return configs.get(query_type, GenerationConfig())

    def _post_process(self, answer: str, context: List[Dict], query_type: QueryType) -> str:
        """后处理生成的答案"""
        if not answer.strip():
            return "抱歉，无法生成有效答案。"

        # 添加引用完整性检查
        answer = self._verify_citations(answer, context)

        # 添加信息来源说明
        source_note = self._generate_source_note(context, query_type)

        return f"{answer}\n\n{source_note}"

    def _verify_citations(self, answer: str, context: List[Dict]) -> str:
        """验证引用的准确性"""
        # 提取答案中的引用
        citation_pattern = r'\[([^,\]]+),\s*(\d{4})\]'
        citations = re.findall(citation_pattern, answer)

        # 验证每个引用是否在上下文中存在
        valid_authors = set()
        valid_years = set()

        for doc in context:
            metadata = doc.get('metadata', {})
            author = metadata.get('作者名称', '')
            year = metadata.get('年份', '')

            if author:
                valid_authors.add(author)
            if year:
                valid_years.add(str(year))

        # 标记可疑引用
        for author, year in citations:
            if author not in valid_authors or year not in valid_years:
                answer = answer.replace(f'[{author}, {year}]', f'[{author}, {year}]*')

        return answer

    def _generate_source_note(self, context: List[Dict], query_type: QueryType) -> str:
        """生成来源说明"""
        doc_count = len(context)
        years = []
        categories = set()

        for doc in context:
            metadata = doc.get('metadata', {})
            year = metadata.get('年份')
            category = metadata.get('分类')

            if year:
                years.append(year)
            if category:
                categories.add(category)

        year_range = f"{min(years)}-{max(years)}" if years else "时间跨度不明"
        category_list = "、".join(list(categories)[:3]) if categories else "多个领域"

        source_note = f"""---
📖 **信息来源说明**
- 参考文献数：{doc_count} 篇
- 时间跨度：{year_range}
- 涉及领域：{category_list}
- 生成时间：{self._get_current_time()}

*注：标有*的引用可能需要进一步核实*"""

        return source_note

    def _generate_fallback(self, query: str, context: List[Dict], query_type: QueryType) -> str:
        """生成降级答案"""
        if not context:
            return "抱歉，没有找到与您的问题相关的文档。请尝试：\n1. 使用更通用的关键词\n2. 检查拼写是否正确\n3. 简化查询内容"

        # 根据查询类型生成结构化的降级答案
        answer = f"关于您的问题「{query}」，虽然无法生成完整答案，但找到以下相关资料供参考：\n\n"

        # 列出相关文档
        for i, doc in enumerate(context[:5], 1):
            metadata = doc.get('metadata', {})
            title = metadata.get('文章名称+副标题', '无标题')
            author = metadata.get('作者名称', '未知作者')
            year = metadata.get('年份', '未知年份')

            answer += f"**[{i}] {title}**\n"
            answer += f"   作者：{author} | 年份：{year}\n"

            # 添加简短摘要
            text = doc.get('text', '')[:200]
            if text:
                answer += f"   摘要：{text}...\n\n"

        # 添加建议
        suggestions = self._get_fallback_suggestions(query_type)
        answer += f"\n💡 **建议**：\n{suggestions}"

        return answer

    def _get_fallback_suggestions(self, query_type: QueryType) -> str:
        """获取降级建议"""
        suggestions = {
            QueryType.COMPARISON: "• 尝试分别搜索每个比较对象\n• 使用更具体的比较维度",
            QueryType.TEMPORAL: "• 尝试搜索特定年代或时期\n• 使用 发展\演变 等关键词",
            QueryType.DEFINITION: "• 查找该概念的上位概念\n• 搜索相关的理论框架",
            QueryType.ANALYTICAL: "• 将问题分解为更小的子问题\n• 寻找相关的案例研究"
        }

        return suggestions.get(query_type, "• 简化查询词\n• 尝试相关概念\n• 查看推荐阅读")

    def _get_current_time(self) -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self.api_key = api_key
        if api_key:
            self._init_llm()

    def is_available(self) -> bool:
        """检查生成功能是否可用"""
        return self.llm is not None


class QueryAnalyzer:
    """查询分析器"""

    def analyze(self, query: str) -> QueryType:
        """分析查询类型"""
        query_lower = query.lower()

        # 定义关键词映射
        type_keywords = {
            QueryType.DEFINITION: ['是什么', '什么是', '定义', '概念', '含义', '解释'],
            QueryType.COMPARISON: ['比较', '对比', '区别', '不同', '相同', '差异', 'vs', '和.*的关系'],
            QueryType.TEMPORAL: ['何时', '什么时候', '历史', '发展', '演变', '起源', '最早', '最新'],
            QueryType.ANALYTICAL: ['为什么', '如何', '怎样', '分析', '评价', '影响', '意义', '作用'],
            QueryType.FACTUAL: ['谁', '哪里', '多少', '几个', '第一', '最大', '最小'],
            QueryType.EXPLORATORY: ['可能', '是否', '有哪些', '还有什么', '其他', '相关']
        }

        # 检查每种类型的关键词
        for query_type, keywords in type_keywords.items():
            for keyword in keywords:
                if re.search(keyword, query_lower):
                    return query_type

        # 默认返回通用类型
        return QueryType.GENERAL


# 便捷函数
def create_generator(api_key: str = None, enable_cot: bool = True) -> AnswerGenerator:
    """创建答案生成器的便捷函数"""
    return AnswerGenerator(api_key=api_key, enable_cot=enable_cot)


# 使用示例
if __name__ == "__main__":
    # 测试代码
    generator = create_generator()

    # 模拟上下文
    test_context = [
        {
            'text': '包豪斯（Bauhaus）是1919年在德国魏玛成立的一所设计学校...',
            'metadata': {
                '年份': 2020,
                '作者名称': '张三',
                '文章名称+副标题': '包豪斯的历史与影响',
                '分类': '设计史'
            },
            'score': 0.95
        }
    ]

    # 测试不同类型的查询
    test_queries = [
        "什么是包豪斯？",
        "比较包豪斯和装饰艺术运动",
        "包豪斯是什么时候成立的？",
        "分析包豪斯对现代设计的影响"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        print("回答: ", generator.generate(query, test_context))
        print("-" * 80)