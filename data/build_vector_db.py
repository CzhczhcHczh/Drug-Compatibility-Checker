"""
将 drug_data.jsonl 中的药品说明书数据切片、嵌入，写入本地向量数据库（ChromaDB），
供后端 LangGraph + LLM 在做"药物配伍冲突分析"时做 RAG 检索使用。

切片策略：
  - 仅向量化与配伍判断相关的字段（白名单 VECTORIZE_FIELDS）
  - 一条药品 → 多个 chunk，每个 chunk = 药名 + 字段名 + 字段内容
  - 长字段按 CHUNK_SIZE 滑窗切（重叠 CHUNK_OVERLAP），避免 embedding 截断

向量库：
  - 使用 chromadb PersistentClient，落盘到 data/chroma_db/
  - 集合名: drug_instructions
  - 距离度量: 余弦相似度

嵌入后端（环境变量 EMBEDDING_BACKEND 切换）：
  - "local"  : sentence-transformers + BAAI/bge-small-zh-v1.5  （默认）
  - "openai" : OpenAI 兼容 Embedding API（OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_EMBED_MODEL）

CLI 用法：
  python build_vector_db.py build              # 全量/增量构建（幂等）
  python build_vector_db.py build --reset      # 清空后重建
  python build_vector_db.py query "阿司匹林禁忌" --top 5
  python build_vector_db.py query "高血压用药" --drug 硝苯地平控释片
  python build_vector_db.py stats              # 集合统计
  python build_vector_db.py reset              # 清空集合
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Iterable, Iterator

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(DATA_DIR, "drug_data.jsonl")
DB_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "drug_instructions"

# 与"配伍/安全/相互作用"判断强相关的字段
VECTORIZE_FIELDS: list[str] = [
    "成份",
    "性状",
    "适应症",
    "用法用量",
    "不良反应",
    "禁忌",
    "注意事项",
    "药物相互作用",
    "孕妇及哺乳期妇女用药",
    "儿童用药",
    "老年用药",
    "药物过量",
    "药理毒理",
    "药代动力学",
]

# 长字段滑窗参数（字符为单位，bge-small-zh 上限 512 token，留余量）
CHUNK_SIZE = 480
CHUNK_OVERLAP = 60

# 入库批大小，避免一次 embedding 太多卡顿
BATCH_SIZE = 64

# bge 系列在做 query 检索时的指令前缀（提升中文检索精度）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


# ===================== 嵌入后端 =====================

class _LocalSTEmbedder:
    """本地 sentence-transformers + BGE 中文模型。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise SystemExit(
                "缺少依赖：sentence-transformers\n"
                "请运行：pip install -r data/requirements.txt"
            ) from e
        print(f"[embed] 加载本地模型: {model_name}（首次会从 HuggingFace 下载）")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if is_query and "bge" in self.model_name.lower():
            texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.tolist()


class _OpenAIEmbedder:
    """OpenAI 兼容 Embedding API（含 DashScope/DeepSeek/智谱兼容端点）。"""

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit(
                "缺少依赖：openai\n请运行：pip install openai"
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise SystemExit("EMBEDDING_BACKEND=openai 需要设置 OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        print(f"[embed] 使用 OpenAI 兼容 API: model={self.model}, base_url={base_url}")

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        # OpenAI Embeddings 不区分 query/passage
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def get_embedder():
    backend = os.environ.get("EMBEDDING_BACKEND", "local").lower()
    if backend == "openai":
        return _OpenAIEmbedder()
    return _LocalSTEmbedder(
        model_name=os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    )


# ===================== 切片 =====================

def _normalize(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\xa0", " ")
        .replace("\u3000", " ")
        .strip()
    )


def _split_long(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


def iter_chunks(records: Iterable[dict]) -> Iterator[dict]:
    """把每条药品记录展开成若干 chunk。"""
    for rec in records:
        drug_name = (rec.get("药品名称") or rec.get("通用名称") or "").strip()
        if not drug_name:
            continue
        generic = (rec.get("通用名称") or "").strip()
        english = (rec.get("英文名称") or "").strip()
        pinyin = (rec.get("汉语拼音") or "").strip()
        url = rec.get("source_url") or ""

        for field in VECTORIZE_FIELDS:
            value = rec.get(field)
            if not value:
                continue
            if isinstance(value, list):
                value = "\n".join(str(v) for v in value)
            value = _normalize(str(value))
            if len(value) < 3:
                continue

            for idx, piece in enumerate(_split_long(value)):
                # chunk 文本：把药名和字段名前置，让 embedding 编码语义角色
                content = f"【药品】{drug_name}\n【字段】{field}\n{piece}"
                doc_id = hashlib.sha1(
                    f"{drug_name}::{field}::{idx}".encode("utf-8")
                ).hexdigest()[:20]
                yield {
                    "id": doc_id,
                    "document": content,
                    "metadata": {
                        "drug_name": drug_name,
                        "generic_name": generic,
                        "english_name": english,
                        "pinyin": pinyin,
                        "field": field,
                        "chunk_index": idx,
                        "source_url": url,
                    },
                }


def load_records(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"找不到数据文件: {path}\n请先运行 data_request.py 抓取数据")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] 跳过坏行: {e}")


# ===================== ChromaDB 操作 =====================

def get_collection(reset: bool = False):
    try:
        import chromadb
    except ImportError as e:
        raise SystemExit(
            "缺少依赖：chromadb\n请运行：pip install -r data/requirements.txt"
        ) from e

    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)

    if reset:
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
            print(f"[db] 已删除集合: {COLLECTION_NAME}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _batched(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def cmd_build(args: argparse.Namespace) -> None:
    client, collection = get_collection(reset=args.reset)
    embedder = get_embedder()

    print(f"[build] 读取: {JSONL_PATH}")
    chunks = list(iter_chunks(load_records(JSONL_PATH)))
    print(f"[build] 切片总数: {len(chunks)}")

    # 增量：跳过已存在的 id
    existing_ids: set[str] = set()
    if not args.reset:
        # collection.get(ids=...) 在 ids 太多时会卡，改成全部 get 一次
        try:
            got = collection.get(include=[])  # 只要 ids
            existing_ids = set(got.get("ids", []))
            print(f"[build] 集合中已有: {len(existing_ids)} 条")
        except Exception as e:
            print(f"[warn] 读取已有 id 失败，按全量构建: {e}")

    pending = [c for c in chunks if c["id"] not in existing_ids]
    print(f"[build] 本次新增: {len(pending)}")
    if not pending:
        print("[build] 无新增内容。")
        return

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):  # type: ignore
            return x

    total = 0
    for batch in tqdm(list(_batched(pending, BATCH_SIZE)), desc="嵌入并写入"):
        ids = [c["id"] for c in batch]
        docs = [c["document"] for c in batch]
        metas = [c["metadata"] for c in batch]
        vectors = embedder.embed(docs, is_query=False)
        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=vectors,
        )
        total += len(batch)

    print(f"[build] 完成。新增/更新 {total} 条；集合现共 {collection.count()} 条。")
    print(f"[build] 数据库路径: {DB_DIR}")


def cmd_query(args: argparse.Namespace) -> None:
    _, collection = get_collection(reset=False)
    embedder = get_embedder()

    qvec = embedder.embed([args.text], is_query=True)
    where: dict | None = None
    conds: list[dict] = []
    if args.drug:
        conds.append({"drug_name": args.drug})
    if args.field:
        conds.append({"field": args.field})
    if conds:
        where = conds[0] if len(conds) == 1 else {"$and": conds}

    res = collection.query(
        query_embeddings=qvec,
        n_results=args.top,
        where=where,
    )

    print(f"\n查询: {args.text}")
    if where:
        print(f"过滤: {where}")
    print("=" * 60)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    if not ids:
        print("（无召回结果）")
        return
    for rank, (i, d, m, dist) in enumerate(zip(ids, docs, metas, dists), 1):
        score = 1.0 - dist  # cosine similarity
        print(f"\n#{rank}  similarity={score:.4f}  drug={m.get('drug_name')}  field={m.get('field')}")
        print(f"    url: {m.get('source_url')}")
        preview = d.replace("\n", " ⏎ ")
        print(f"    {preview[:240]}{'…' if len(preview) > 240 else ''}")


def cmd_stats(args: argparse.Namespace) -> None:
    _, collection = get_collection(reset=False)
    n = collection.count()
    print(f"集合: {COLLECTION_NAME}")
    print(f"路径: {DB_DIR}")
    print(f"总条目: {n}")
    if n == 0:
        return
    sample = collection.get(limit=min(n, 5000), include=["metadatas"])
    metas = sample.get("metadatas") or []
    drugs = {m.get("drug_name") for m in metas if m}
    from collections import Counter
    fields = Counter(m.get("field") for m in metas if m)
    print(f"覆盖药品（采样上限 5000）: ~{len(drugs)} 种")
    print("字段分布（采样）:")
    for k, v in fields.most_common():
        print(f"  {v:6d}  {k}")


def cmd_reset(args: argparse.Namespace) -> None:
    _, collection = get_collection(reset=True)
    print(f"已清空集合，当前条目: {collection.count()}")


# ===================== CLI =====================

def main() -> None:
    parser = argparse.ArgumentParser(description="药品说明书向量库构建/查询工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="构建/增量更新向量库")
    p_build.add_argument("--reset", action="store_true", help="清空集合后重建")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="向量检索测试")
    p_query.add_argument("text", help="自然语言查询")
    p_query.add_argument("--top", type=int, default=5, help="返回前 N 条（默认 5）")
    p_query.add_argument("--drug", default=None, help="按药品名精确过滤")
    p_query.add_argument("--field", default=None, help="按字段过滤，如 禁忌 / 药物相互作用")
    p_query.set_defaults(func=cmd_query)

    p_stats = sub.add_parser("stats", help="查看集合统计")
    p_stats.set_defaults(func=cmd_stats)

    p_reset = sub.add_parser("reset", help="清空集合")
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
