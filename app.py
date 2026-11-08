# -*- coding: utf-8 -*-
"""
校园生活助手 Agent —— 大学校园一站式服务智能体
=================================================
功能：
  1. 问答校园制度（基于上传的规章制度 PDF 知识库检索）
  2. 生成校园办事流程（内置常见办事流程模板）
  3. 根据学生需求给出行动建议（意图识别 + 规则匹配）

技术特点：
  - 纯本地运行，不依赖大模型 API，无需密钥、无需联网
  - 检索采用纯 Python 实现的 TF-IDF + 余弦相似度，无重型依赖
  - PDF 解析使用 pdfplumber，按段落分块建立知识库
"""

import os
import re
import math
import uuid
from collections import Counter

import requests
from flask import Flask, request, jsonify, send_from_directory

import pdfplumber

# ---------------------------------------------------------------------------
# 配置加载：优先环境变量，其次 config.json
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_config():
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                import json as _json
                cfg = _json.load(fh) or {}
        except Exception:
            cfg = {}
    return cfg


CONFIG = _load_config()

# LLM 配置（兼容 DeepSeek / 火山方舟等 OpenAI 格式平台）
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ARK_API_KEY") or CONFIG.get("api_key", "")
LLM_BASE_URL = CONFIG.get("base_url", "https://api.deepseek.com")
LLM_MODEL = CONFIG.get("model", "deepseek-chat")
LLM_ENABLED = bool(LLM_API_KEY)

# ---------------------------------------------------------------------------
# 应用初始化
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 限制上传 30MB


# ---------------------------------------------------------------------------
# 一、知识库（全局状态，内存中维护）
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """内存知识库：保存 PDF 解析出的文本块，并支持 TF-IDF 检索。"""

    def __init__(self):
        self.documents = []        # list[dict]: {id, source, text, title}
        self.chunks = []           # list[dict]: {id, doc_id, text}
        self._idf = {}             # 词的 IDF
        self._doc_tf = []          # 每个 chunk 的词频向量

    # ---- 文本分块 ----
    @staticmethod
    def _split_text(text, max_len=300, overlap=30):
        """把长文本切分成知识块。

        优先按「第X条/第X章」切分（适合规章制度类文档，让每一条成为独立
        知识块，检索更精准）；若无此类结构，则按句子边界切分成带重叠的块。
        """
        text = text.strip()
        # 1) 尝试按「第X条/第X章/第X节」切分
        #    匹配形如「第三条」「第十二条」「第一章」等，含中文数字/阿拉伯数字
        pat = r"(?=第[一二三四五六七八九十百零0-9０-９]+[条章节])"
        parts = re.split(pat, text)
        parts = [p.strip() for p in parts if p.strip()]
        # 如果切出了多个以「第X条/章」开头的段落，说明条文结构成立
        structured = [p for p in parts if re.match(r"^第[一二三四五六七八九十百零0-9０-９]+[条章节]", p)]
        if len(structured) >= 3:
            # 存在清晰条文结构：把每个「第X条/章」段作为一块；
            # 过长的条（如含多个子项）保持原样，不做二次切分，保证完整性
            chunks = []
            for p in parts:
                # 去掉段内多余空白，保留换行
                p = re.sub(r"[ \t\u3000]+", " ", p).strip()
                if len(p) >= 6:
                    chunks.append(p)
            # 若某块过长（>2*max_len），再按句子二次切分
            final = []
            for c in chunks:
                if len(c) > max_len * 2:
                    final.extend(KnowledgeBase._split_by_sentence(c, max_len, overlap))
                else:
                    final.append(c)
            return final

        # 2) 无条文结构：按句子切分
        return KnowledgeBase._split_by_sentence(text, max_len, overlap)

    @staticmethod
    def _split_by_sentence(text, max_len=300, overlap=30):
        """按句子边界切分成带重叠的块。"""
        sentences = re.split(r"(?<=[。！？!?；;])", text)
        chunks, cur = [], ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(cur) + len(s) > max_len and cur:
                chunks.append(cur)
                cur = cur[-overlap:] if overlap else ""  # 保留重叠
            cur += s
        if cur.strip():
            chunks.append(cur.strip())
        return [c for c in chunks if len(c.strip()) >= 8]

    # ---- 添加文档 ----
    def add_pdf(self, pdf_bytes, filename):
        """解析 PDF 字节流，分块加入知识库，返回 (doc_id, chunk_count)。"""
        full_text = self._extract_pdf_text(pdf_bytes)
        if not full_text or len(full_text.strip()) < 20:
            raise ValueError("未能从该 PDF 中提取到有效文本（可能是扫描版/图片型 PDF）")
        return self.add_text(full_text, filename)

    def add_text(self, text, filename):
        """直接添加纯文本到知识库，返回 (doc_id, chunk_count)。"""
        doc_id = uuid.uuid4().hex
        chunks = self._split_text(text)
        if not chunks:
            raise ValueError("文本太短，无法切分成有效知识块")

        doc = {"id": doc_id, "source": filename, "text": text,
               "title": filename}
        self.documents.append(doc)
        for c in chunks:
            self.chunks.append({"id": uuid.uuid4().hex, "doc_id": doc_id, "text": c})
        self._rebuild_index()
        return doc_id, len(chunks)

    @staticmethod
    def _extract_pdf_text(pdf_bytes):
        """用 pdfplumber 抽取全部页文本。"""
        import io
        parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
        text = "\n".join(parts)
        # 合并被 PDF 硬换行打断的中文行（中文句子内不应有孤立换行）
        # 规则：若上一行以中文字符结尾、下一行以中文字符开头，则合并为一行
        lines = text.split("\n")
        merged = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if merged and merged[-1] and re.search(r"[\u4e00-\u9fa5]$", merged[-1]) \
                    and re.match(r"^[\u4e00-\u9fa5]", s):
                merged[-1] = merged[-1] + s
            else:
                merged.append(s)
        return "\n".join(merged)

    # ---- 索引重建 ----
    def _rebuild_index(self):
        """重建 TF-IDF 索引。"""
        docs_tokens = [self._tokenize(c["text"]) for c in self.chunks]
        n = len(docs_tokens)
        df = Counter()
        for tokens in docs_tokens:
            for w in set(tokens):
                df[w] += 1

        self._idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
        self._doc_tf = []
        for tokens in docs_tokens:
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec = {w: (f / total) * self._idf[w] for w, f in tf.items()}
            self._doc_tf.append(vec)

    @staticmethod
    def _tokenize(text):
        """中文分词（简单版）：按非中文/非字母数字切分，并做二元组增强。"""
        text = text.lower()
        # 提取中文连续串、英文单词、数字
        tokens = re.findall(r"[\u4e00-\u9fa5]|[a-z0-9]+", text)
        # 中文二元组（bigram），提升检索召回
        bigrams = []
        chinese = re.findall(r"[\u4e00-\u9fa5]", text)
        for i in range(len(chinese) - 1):
            bigrams.append(chinese[i] + chinese[i + 1])
        return tokens + bigrams

    # ---- 检索 ----
    def search(self, query, top_k=3):
        """返回与 query 最相似的 top_k 个文本块及其相似度。"""
        if not self.chunks:
            return []
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        q_tf = Counter(q_tokens)
        total = len(q_tokens) or 1
        q_vec = {}
        for w, f in q_tf.items():
            if w in self._idf:
                q_vec[w] = (f / total) * self._idf[w]

        scores = []
        for i, dvec in enumerate(self._doc_tf):
            sim = self._cosine(q_vec, dvec)
            scores.append((sim, self.chunks[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [(s, c) for s, c in scores[:top_k] if s > 0.0]

    @staticmethod
    def _cosine(a, b):
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def status(self):
        return {
            "doc_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "documents": [{"id": d["id"], "title": d["title"]} for d in self.documents],
        }


KB = KnowledgeBase()


# ---------------------------------------------------------------------------
# 二、内置校园办事流程模板库
# ---------------------------------------------------------------------------
# 每条流程：关键词列表（用于意图识别）、标题、步骤、提示
PROCESS_LIBRARY = [
    {
        "id": "canteen",
        "keywords": ["食堂", "饭卡", "校园卡", "充卡", "饭堂", "就餐", "一卡通", "餐厅", "吃饭"],
        "title": "食堂就餐与校园卡充值流程",
        "steps": [
            "前往校内任一食堂（如第一食堂、第二食堂）",
            "使用校园卡（一卡通）刷卡支付，或使用微信/支付宝扫码",
            "校园卡余额不足时，到食堂充值窗口或线上校园卡 APP 充值",
            "校园卡丢失请立即到卡务中心挂失，避免余额损失",
        ],
        "tip": "建议错峰就餐（11:30 前或 12:30 后），避开高峰期排队。",
    },
    {
        "id": "course",
        "keywords": ["选课", "退课", "学分", "课程", "选修", "必修", "课表", "抢课", "教务"],
        "title": "选课与退课流程",
        "steps": [
            "在规定选课时间内登录教务处/教务系统",
            "进入「网上选课」模块，查看培养方案与可选课程",
            "必修课优先选择，选修课按个人兴趣与学分要求选择",
            "退课需在退补选开放时间内操作，逾期将无法退课",
        ],
        "tip": "热门课程名额有限，建议提前确认选课时间并准点抢课；注意学分上限。",
    },
    {
        "id": "repair",
        "keywords": ["报修", "维修", "水电", "空调", "灯坏", "漏水", "故障", "宿舍", "后勤"],
        "title": "宿舍报修流程",
        "steps": [
            "登录后勤报修平台（或校园 APP 的「报修」入口）",
            "填写报修信息：楼栋、房间号、故障类型、故障描述、联系电话",
            "提交后等待维修师傅接单，一般 1-2 个工作日内上门",
            "维修完成后在平台上确认并评价",
        ],
        "tip": "紧急情况（如漏水、断电）建议同时拨打后勤值班电话，可更快响应。",
    },
    {
        "id": "scholarship",
        "keywords": ["奖学金", "助学金", "贫困", "资助", "助学贷款", "评定", "家庭经济"],
        "title": "奖学金与助学金申请流程",
        "steps": [
            "关注学生处/资助中心发布的评定通知",
            "按要求准备申请材料（申请表、成绩证明、家庭经济情况说明等）",
            "在规定时间内在资助系统提交申请",
            "参加班级/学院民主评议，等待审核公示",
        ],
        "tip": "奖学金评定通常综合学业成绩与综合素质，提前了解评定细则更有把握。",
    },
    {
        "id": "library",
        "keywords": ["图书馆", "借书", "还书", "自习", "占座", "文献", "续借"],
        "title": "图书馆借还书流程",
        "steps": [
            "凭校园卡进入图书馆",
            "在检索机或官网查询馆藏位置",
            "到书架取书后，在自助借还机或服务台办理借阅",
            "按期归还，超期可能产生罚款；可在官网续借",
        ],
        "tip": "考试周图书馆座位紧张，可通过座位预约系统提前预约。",
    },
    {
        "id": "registration",
        "keywords": ["报到", "注册", "新生", "入学", "住宿", "体检", "军训"],
        "title": "新生报到流程",
        "steps": [
            "携带录取通知书、身份证等到指定报到点",
            "办理报到注册，领取校园卡与宿舍钥匙",
            "办理住宿登记，入住宿舍",
            "参加新生体检、入学教育等后续安排",
        ],
        "tip": "报到当天人流量大，建议按学院通知分批到校，提前准备好证件材料。",
    },
    {
        "id": "leave",
        "keywords": ["请假", "假条", "旷课", "请假流程", "事假", "病假"],
        "title": "学生请假流程",
        "steps": [
            "向辅导员说明请假事由并领取/填写请假条",
            "按请假天数逐级审批（辅导员 → 学院）",
            "审批通过后留存假条，及时告知任课老师",
            "病假建议附上医院诊断证明",
        ],
        "tip": "请提前请假，避免旷课记录影响平时成绩与综合素质评价。",
    },
    {
        "id": "network",
        "keywords": ["校园网", "上网", "wifi", "无线", "断网", "网络账号", "充值网费"],
        "title": "校园网开通与报障流程",
        "steps": [
            "使用学号激活校园网账号，设置密码",
            "连接校园无线网（SSID 通常为 campus 或学校简称）",
            "网费不足时在自助平台或充值点充值",
            "网络故障可拨打网络中心报障电话或在线提交工单",
        ],
        "tip": "记住学号密码，勿外借账号，避免产生违规记录。",
    },
]

# 通用兜底流程
FALLBACK_PROCESS = {
    "id": "general",
    "title": "校园办事通用流程",
    "steps": [
        "明确你要办理的事项类型（如证明、报销、活动申请等）",
        "前往对应职能部门（教务处/学生处/后勤/财务等）官网或窗口咨询",
        "按要求准备并提交材料",
        "跟进办理进度，及时领取结果",
    ],
    "tip": "多数业务可先在学校官网或校园 APP 的「办事大厅」查找线上入口。",
}


# ---------------------------------------------------------------------------
# 三、意图识别与建议生成
# ---------------------------------------------------------------------------
class LLMClient:
    """大模型客户端（兼容 OpenAI 格式），支持 DeepSeek / 火山方舟等，带超时与降级。"""

    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _endpoint(self):
        """构造 chat/completions 完整地址，兼容多种 base_url 写法。

        - DeepSeek:  base_url = https://api.deepseek.com
                      -> https://api.deepseek.com/v1/chat/completions
        - 火山方舟:  base_url = https://ark.cn-beijing.volces.com/api/v3
                      -> https://ark.cn-beijing.volces.com/api/v3/chat/completions
        """
        if re.search(r"/v\d+$", self.base_url):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def chat(self, system_prompt, user_prompt, temperature=0.3, max_tokens=1200):
        """调用大模型，成功返回文本，失败抛出异常（由上层降级处理）。"""
        url = self._endpoint()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            # 尝试解析错误信息
            try:
                err = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err = resp.text
            raise RuntimeError(f"大模型返回 {resp.status_code}：{err}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


LLM = LLMClient(LLM_API_KEY, LLM_BASE_URL, LLM_MODEL) if LLM_ENABLED else None

SYSTEM_PROMPT = (
    "你是一名校园生活助手，服务对象是在校大学生。"
    "你的职责是根据提供的【参考材料】准确回答学生关于校园规章制度、办事流程的问题。"
    "回答要求：\n"
    "1. 优先严格依据参考材料中的原文内容作答，不要编造材料里没有的规定。\n"
    "2. 语言自然、友好、简洁，用通俗的话把规定讲清楚，可适当分点。\n"
    "3. 如果参考材料不足以回答，要如实说明，并给出一般性建议，不要杜撰。\n"
    "4. 涉及处分、学籍等重要事项时，提醒学生以学校官方文件为准。"
)


def build_llm_prompt(query, kb_hits, proc):
    """把检索结果与流程模板拼成给大模型的上下文。"""
    sections = []
    if kb_hits:
        refs = []
        for sim, c in kb_hits:
            doc = next((d for d in KB.documents if d["id"] == c["doc_id"]), None)
            src = doc["title"] if doc else "知识库"
            refs.append(f"【来源：《{src}》】\n{c['text'].strip()}")
        sections.append("【参考材料——规章制度原文】\n" + "\n\n".join(refs))
    if proc:
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(proc["steps"], 1))
        sections.append(
            f"【参考材料——办事流程模板】\n{proc['title']}\n{steps}\n建议：{proc['tip']}"
        )
    context = "\n\n".join(sections) if sections else "（暂无匹配的参考材料）"
    return (
        f"学生提问：{query}\n\n"
        f"{context}\n\n"
        f"请根据以上参考材料，用自然语言回答学生的问题。"
    )


def detect_intent(query):
    """根据关键词命中情况，返回匹配的流程模板（及命中得分）。"""
    best, best_score = None, 0
    for proc in PROCESS_LIBRARY:
        score = sum(1 for kw in proc["keywords"] if kw in query)
        if score > best_score:
            best, best_score = proc, score
    return best, best_score


def build_answer(query):
    """核心：检索知识库 + 识别意图，交给大模型生成自然回答（无 Key 时降级为规则回答）。"""
    # 1) 知识库检索（如果已上传文档）
    kb_hits = KB.search(query, top_k=3)

    # 2) 流程模板意图识别
    proc, proc_score = detect_intent(query)

    sources = [{"sim": round(s, 4), "title": next(
        (d["title"] for d in KB.documents if d["id"] == c["doc_id"]), "知识库")}
        for s, c in kb_hits]

    # 3) 若已配置大模型，优先用大模型生成自然回答
    if LLM is not None:
        try:
            user_prompt = build_llm_prompt(query, kb_hits, proc if proc_score > 0 else None)
            llm_text = LLM.chat(SYSTEM_PROMPT, user_prompt)
            return {
                "text": llm_text,
                "intent": proc["id"] if (proc and proc_score > 0) else None,
                "intent_title": proc["title"] if (proc and proc_score > 0) else None,
                "kb_used": bool(kb_hits),
                "llm_used": True,
                "sources": sources,
            }
        except Exception as e:
            # 大模型调用失败，降级到规则回答，并在回答里不暴露内部错误
            print(f"[LLM 调用失败，降级为规则回答] {e}")

    # 4) 降级：规则 + 检索 拼接回答（原逻辑）
    answer_parts = []
    if kb_hits:
        best_sim, best_chunk = kb_hits[0]
        doc = next((d for d in KB.documents if d["id"] == best_chunk["doc_id"]), None)
        source = doc["title"] if doc else "知识库"
        answer_parts.append(
            f"根据你上传的《{source}》中的相关规定：\n\n"
            f"{best_chunk['text'].strip()}"
        )
        for sim, c in kb_hits[1:]:
            if sim > 0.15:
                answer_parts.append(f"（补充）{c['text'].strip()}")

    if proc and proc_score > 0:
        answer_parts.append(
            f"\n\n【{proc['title']}】\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(proc["steps"], 1))
            + f"\n\n💡 建议：{proc['tip']}"
        )
    elif not kb_hits:
        answer_parts.append(
            "我暂时没有在你的知识库或内置流程库中找到相关答案。\n\n"
            "你可以：\n"
            "1. 上传本校的规章制度文档，我会据此作答；\n"
            "2. 尝试换一种问法（例如「食堂怎么充值」「怎么报修」「奖学金怎么申请」等）；\n"
            "3. 询问以下常见事项：选课、报修、食堂、奖学金、图书馆、请假、校园网、新生报到等。"
        )

    text = "\n".join(p for p in answer_parts if p)
    return {
        "text": text,
        "intent": proc["id"] if (proc and proc_score > 0) else None,
        "intent_title": proc["title"] if (proc and proc_score > 0) else None,
        "kb_used": bool(kb_hits),
        "llm_used": False,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# 四、Flask 路由
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "问题不能为空"}), 400
    result = build_answer(query)
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "未收到文件"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "文件名为空"}), 400

    name = f.filename
    ext = os.path.splitext(name)[1].lower()
    ALLOWED = {".pdf", ".txt", ".doc", ".docx"}

    if ext not in ALLOWED:
        return jsonify({"error": f"仅支持 {', '.join(sorted(ALLOWED))} 格式"}), 400

    try:
        data = f.read()
        if ext == ".pdf":
            doc_id, chunk_count = KB.add_pdf(data, name)
        elif ext in (".doc", ".docx"):
            text = _extract_word_text(data, name)
            if not text or len(text.strip()) < 20:
                return jsonify({"error": "未能从该 Word 文档中提取到有效文本"}), 400
            doc_id, chunk_count = KB.add_text(text, name)
        else:  # .txt
            text = _decode_text(data)
            doc_id, chunk_count = KB.add_text(text, name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"解析失败：{e}"}), 500

    return jsonify({
        "ok": True,
        "doc_id": doc_id,
        "filename": name,
        "chunk_count": chunk_count,
        "status": KB.status(),
    })


def _decode_text(data):
    """尽力解码文本字节（支持 utf-8 / gbk / utf-16）。"""
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="ignore")


def _extract_word_text(data, filename):
    """从 .doc/.docx 字节流中提取纯文本。

    .docx 用标准 zip+xml 解析；.doc（老式二进制）在 Windows 上用 Word COM
    转换，否则尝试简单的 UTF-16LE 提取。
    """
    name = filename.lower()
    if name.endswith(".docx"):
        return _extract_docx_text(data)

    # .doc 老格式
    text = _extract_legacy_doc_text(data)
    if text and len(text.strip()) >= 20:
        return text
    # 降级：尝试用 Word COM（仅在 Windows 且安装了 Word 时）
    return _extract_doc_via_com(data, filename)


def _extract_docx_text(data):
    """解析 .docx（zip 容器内的 document.xml）。"""
    import zipfile
    import io as _io
    from xml.etree import ElementTree as ET
    with zipfile.ZipFile(_io.BytesIO(data)) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for p in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        texts = [t.text for t in p.iter(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
            if t.text]
        paragraphs.append("".join(texts))
    return "\n".join(paragraphs)


def _extract_legacy_doc_text(data):
    """从老式 .doc 二进制中尽力提取 UTF-16LE 存储的中文正文。"""
    import re
    text = data.decode("utf-16-le", errors="ignore")
    # 提取连续的中文段落（含常见标点与数字）
    chunks = re.findall(
        r"[\u4e00-\u9fa5][\u4e00-\u9fa5，。、；：（）《》【】\d\s·—\-、‘’“”]{3,}",
        text)
    return "\n".join(c.strip() for c in chunks)


def _extract_doc_via_com(data, filename):
    """在 Windows 上通过 Word COM 把 .doc 转成纯文本（降级方案）。"""
    import tempfile
    import subprocess
    import sys

    # 写到临时文件
    tmp_dir = tempfile.mkdtemp()
    src = os.path.join(tmp_dir, filename)
    dst = os.path.join(tmp_dir, "out.txt")
    with open(src, "wb") as fh:
        fh.write(data)

    # 用 PowerShell 调 Word COM 转换
    ps_script = (
        "$ErrorActionPreference='Stop';"
        f"$w=New-Object -ComObject Word.Application;"
        "$w.Visible=$false;"
        f"$d=$w.Documents.Open('{src}', $false, $true);"
        f"$d.SaveAs2('{dst}', 7);"
        "$d.Close($false);$w.Quit();"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, timeout=60)
        if os.path.exists(dst):
            with open(dst, "rb") as fh:
                return _decode_text(fh.read())
    except Exception:
        pass
    return ""


@app.route("/api/kb", methods=["GET"])
def api_kb():
    return jsonify(KB.status())


@app.route("/api/status", methods=["GET"])
def api_status():
    """返回服务状态，含大模型是否启用。"""
    return jsonify({
        "llm_enabled": LLM_ENABLED,
        "llm_model": LLM_MODEL if LLM_ENABLED else None,
        "kb": KB.status(),
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """清空知识库（可选）。"""
    KB.documents.clear()
    KB.chunks.clear()
    KB._idf.clear()
    KB._doc_tf.clear()
    return jsonify({"ok": True, "status": KB.status()})


if __name__ == "__main__":
    print("=" * 50)
    print("  校园生活助手 Agent 已启动")
    print("  请在浏览器打开： http://127.0.0.1:5000")
    if LLM_ENABLED:
        print(f"  大模型已启用：{LLM_MODEL} ({LLM_BASE_URL})")
    else:
        print("  大模型未启用（未配置 API Key），使用规则+检索模式")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=False)
