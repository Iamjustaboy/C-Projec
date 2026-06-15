# 部署说明

本文档说明如何把企业知识库 RAG 助手作为一个可交付 Demo 部署运行。

## 1. 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填入：

```text
OPENAI_API_KEY=你的 OpenAI API Key
```

启动应用：

```bash
streamlit run app.py
```

浏览器打开 Streamlit 输出的地址。首次提问或上传文档后，系统会构建 `vectorstore/` 本地向量库。

## 2. Docker 部署

构建镜像：

```bash
docker build -t enterprise-rag-assistant .
```

运行容器：

```bash
docker run --rm -p 8501:8501 \
  -e OPENAI_API_KEY=你的 OpenAI API Key \
  enterprise-rag-assistant
```

访问：

```text
http://localhost:8501
```

## 3. 持久化上传文档和向量库

默认情况下，容器重启后上传文件和向量库会丢失。部署时建议挂载两个目录：

```bash
docker run --rm -p 8501:8501 \
  -e OPENAI_API_KEY=你的 OpenAI API Key \
  -v $(pwd)/data/uploaded_docs:/app/data/uploaded_docs \
  -v $(pwd)/vectorstore:/app/vectorstore \
  enterprise-rag-assistant
```

这样用户上传的文档和 Chroma 本地索引都会保存在宿主机。

## 4. 环境变量

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI API Key | 无 |
| `OPENAI_MODEL` | 回答模型 | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding 模型 | `text-embedding-3-small` |
| `RAG_TOP_K` | 每次检索召回片段数量 | `4` |
| `RAG_CHUNK_SIZE` | 文档切分长度 | `900` |
| `RAG_CHUNK_OVERLAP` | 相邻片段重叠长度 | `160` |
| `RAG_UPLOAD_DIR` | 用户上传文档目录 | `data/uploaded_docs` |
| `RAG_VECTORSTORE_DIR` | Chroma 索引目录 | `vectorstore` |

## 5. 生产化注意事项

- 不要把 `.env`、`vectorstore/`、用户上传文档提交到 Git。
- 如果部署到公网，需要增加登录鉴权和上传文件大小限制。
- PDF 解析依赖文本层；扫描版 PDF 需要额外接 OCR。
- 企业真实场景应加入文档权限过滤，避免用户检索到无权限资料。
