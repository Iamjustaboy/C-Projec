# RAG 项目学习注释

这份说明用于帮助你从零理解当前项目骨架。可以把它当成阅读代码时的路线图。

## 1. 一次完整问答发生了什么

```text
用户问题
  -> 把问题交给 Chroma 检索器
  -> Chroma 用 OpenAI Embeddings 把问题转成向量
  -> 在本地向量库中找 Top-K 个最相似文档片段
  -> 把片段内容和 source/chunk_id 组装进 Prompt
  -> ChatOpenAI 基于检索资料生成回答
  -> Streamlit 展示回答和引用来源
```

对应代码：

- 页面入口：`app.py`
- 问答链路：`src/rag/chain.py`
- Prompt 规则：`src/rag/prompts.py`

## 2. 为什么要先构建向量库

大模型本身不会自动读取你的本地文档。RAG 需要先把企业文档变成可检索的向量索引：

```text
data/sample_docs/*.md
data/uploaded_docs/*.{txt,md,pdf}
  -> load_documents()
  -> split_documents()
  -> OpenAIEmbeddings
  -> Chroma.from_documents()
  -> vectorstore/
```

对应代码：`src/rag/index.py`。

`vectorstore/` 是本地索引目录。第一次提问、点击“重建知识库索引”或上传文档后会生成它，后续可以直接加载，避免每次启动都重新向量化。

## 3. Document 的 metadata 为什么重要

每个文档片段除了正文，还会带上：

- `source`：来源文件路径。
- `filename`：文件名。
- `file_type`：文档类型。
- `chunk_id`：切分后的片段编号。

这些信息用于回答后的“引用来源”展示。没有 metadata，用户就只能看到答案，却不知道答案来自哪份资料。

对应代码：`src/rag/loaders.py`。

## 4. chunk_size 和 chunk_overlap 怎么理解

`chunk_size` 控制每个片段大约多长。片段太短会丢上下文，片段太长会让检索不精准。

`chunk_overlap` 控制相邻片段之间保留多少重复内容。它可以减少一句话或一个流程被切断带来的语义损失。

当前默认值在 `.env.example` 中：

```text
RAG_CHUNK_SIZE=900
RAG_CHUNK_OVERLAP=160
```

## 5. Prompt 为什么要求“不知道”

RAG 系统也可能检索不到正确资料。如果 Prompt 不约束模型，模型可能会凭常识编造答案。

本项目在 `SYSTEM_PROMPT` 中明确要求：

- 只能根据检索资料回答。
- 资料中没有答案时，说“根据当前知识库资料，我不知道答案。”
- 涉及数字、流程、时间、责任人时必须与资料一致。

对应代码：`src/rag/prompts.py`。

## 6. Top-K 应该怎么调

Top-K 表示每次提问召回几个文档片段：

- 较小：答案更聚焦，token 成本低，但可能漏掉资料。
- 较大：资料更全，但可能引入无关内容，让模型回答变散。

当前页面在侧边栏提供 2 到 8 的滑杆，默认值来自 `.env` 的 `RAG_TOP_K`。

## 7. 你可以尝试的学习任务

1. 在页面上传一份 `.txt`、`.md` 或 `.pdf` 文档，然后点击“保存上传并重建索引”。
2. 修改 `RAG_TOP_K`，观察引用来源数量和回答质量变化。
3. 修改 `SYSTEM_PROMPT`，观察模型回答风格和拒答行为变化。
4. 在 `load_documents()` 中继续扩展 `.docx` 或网页加载逻辑。
5. 给 `answer_question()` 增加相似度分数展示，帮助判断检索质量。

## 8. 上传功能怎么接入 RAG

上传功能在 `app.py` 中完成两件事：

1. 使用 `st.file_uploader()` 接收 `.txt`、`.md`、`.pdf` 文件。
2. 保存到 `data/uploaded_docs/` 后调用 `build_vector_store(..., force_rebuild=True)`。

真正读取上传文件的逻辑仍然在 `src/rag/loaders.py`，所以页面层不需要知道 PDF 如何解析。这样做的好处是：以后把前端换成 API 服务时，加载和建库逻辑仍然可以复用。

上传目录默认被 `.gitignore` 忽略，因为真实企业文档通常不应该提交到代码仓库。
