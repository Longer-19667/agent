# 校园生活助手 Agent —— 使用说明

大学校园一站式服务智能体。支持接入 DeepSeek 大模型（可选），无 Key 时自动降级为本地规则模式，均可离线运行。

## 一、功能

1. **问答校园制度**：上传本校规章制度文档后，基于文件内容检索作答。支持 **PDF、Word（.doc / .docx）、纯文本（.txt）** 四种格式。
2. **生成校园办事流程**：内置选课、报修、食堂、奖学金、图书馆、请假、校园网、新生报到等流程模板。
3. **给出行动建议**：识别学生问题意图，给出可执行的操作步骤与提示。
4. **条文级精准检索**：自动识别「第X条 / 第X章」结构，把规章制度按条切分，问答时精准定位到具体条款（而非整篇返回）。
5. **大模型自然回答（可选）**：接入 DeepSeek 后，由大模型依据检索到的条款原文，用自然语言组织回答；调用失败或未配置时自动降级为规则回答。

## 二、环境准备（一次性）

本项目依赖 `flask`、`pdfplumber`、`requests` 三个 Python 库。

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ flask pdfplumber requests
```

> 说明：国内直连 PyPI 官方源通常失败，已实测阿里云镜像可用。
> 若在 JoyCode 的终端里安装，命令同上。

## 三、配置大模型（可选，强烈推荐）

在项目根目录的 `config.json` 中填入你的大模型 API Key 和模型 ID。已兼容 **DeepSeek** 与 **火山方舟**（及其他 OpenAI 格式平台）。

### 火山方舟（豆包）示例

```json
{
  "api_key": "你的方舟 API Key",
  "base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "model": "doubao-seed-2-1-pro-260915"
}
```

> 方舟的 `model` 需填你在控制台开通的模型 ID，或创建的「推理接入点」ID（形如 `ep-xxxxx`）。

### DeepSeek 示例

```json
{
  "api_key": "sk-你的key",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-flash"
}
```

> 可用模型可在 DeepSeek 控制台或调用 `/v1/models` 查看（当前账号可用 `deepseek-flash`、`deepseek-v4-pro`）。

- 也可用环境变量 `DEEPSEEK_API_KEY`（或 `ARK_API_KEY`）代替写入文件。
- 不填 Key 也能用，会自动退回「规则 + 检索」模式。
- `config.json` 已加入 `.gitignore`，不会随代码泄露。

## 四、启动

```bash
cd "C:\Users\圣主\Desktop\区块链1\校园生活助手Agent"
python app.py
```

启动成功后，浏览器打开：

```
http://127.0.0.1:5000
```

启动日志会提示「大模型已启用」或「大模型未启用」。

## 五、使用

1. 点右上角「上传规章制度文档」，选择本校的规章制度文件（PDF / Word / 文本均可）。
2. 在输入框提问，或点下方快捷按钮（食堂充值 / 宿舍报修 / 选课 / 奖学金 / 新生报到）。
3. 顶部状态栏会显示当前是「AI 模式」还是「规则模式」。

## 六、接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | 前端聊天页面 |
| POST | `/api/chat` | 问答（JSON：`{"query": "..."}`） |
| POST | `/api/upload` | 上传文档（表单字段 `file`，支持 .pdf/.doc/.docx/.txt） |
| GET  | `/api/kb` | 知识库状态 |
| GET  | `/api/status` | 服务状态（含大模型是否启用） |
| POST | `/api/reset` | 清空知识库 |

## 七、项目结构

```
校园生活助手Agent/
├── app.py              # 后端（Flask + 文档解析 + 检索 + 大模型接入 + 流程模板）
├── static/
│   └── index.html      # 前端聊天界面
├── config.json         # 大模型配置（含 API Key，勿提交）
├── .gitignore          # 忽略敏感文件
├── requirements.txt    # 依赖清单
└── README.md           # 本说明
```

## 八、技术说明（检索原理）

- **文档解析**：
  - PDF：`pdfplumber` 逐页抽取文本。
  - Word（.docx）：解析 zip 容器内的 `document.xml`。
  - Word（.doc 老格式）：UTF-16LE 提取中文正文；在 Windows 上可降级用 Word COM 转换。
  - 纯文本：自动识别 utf-8 / gbk / utf-16 编码。
- **条文切分**：自动识别「第X条 / 第X章 / 第X节」，把规章制度按条切成独立知识块，问答时精准定位到具体条款；无条文结构的文档则按句子切分。
- **检索**：纯 Python 实现 TF-IDF 权重 + 余弦相似度。中文采用「单字 + 二元组（bigram）」分词，无需 jieba 等额外依赖。
- **意图识别**：内置流程模板库，按关键词命中情况匹配意图，给出对应办事流程。
- **回答策略**：
  - 配置了大模型：检索到的条款原文 + 流程模板作为上下文，交给 DeepSeek 生成自然回答；失败自动降级。
  - 未配置：知识库检索结果 + 流程建议拼接回答。

## 九、局限与可扩展方向

- 大模型回答的准确性依赖检索质量；涉及处分、学籍等重大事项，已在提示词中要求以官方文件为准。
- 扫描版（图片型）PDF 无法直接提取文字，需要 OCR（可后续接入 `paddleocr` 等）。
- 知识库当前存内存，重启后清空；如需持久化，可把 chunks 存入 SQLite 或文件。
- 需要多轮对话记忆时，可扩展保存历史消息到会话中。
