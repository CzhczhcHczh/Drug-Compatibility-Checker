"""
药源网（yaopinnet.com）化学药说明书抓取脚本。

站点结构（实测）：
  入口      : https://www.yaopinnet.com/huayao1/                   仅含字母导航
  字母页    : https://www.yaopinnet.com/huayao1/<a-z>1.htm          列出该字母下的详情链接
  详情页    : https://www.yaopinnet.com/huayao/hy<digits>[a-z]?.htm 真正的说明书
  字段载体  : <ul><li class="smsli">【字段名】内容</li>...</ul>     字段值用 <br> 分行

输出：data/drug_data.jsonl（每行一个 JSON 对象，便于断点续抓）
"""

import json
import os
import random
import re
import time
from urllib.parse import urljoin

import requests
from lxml import etree
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.yaopinnet.com"
ENTRY = f"{BASE}/huayao1/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Referer": BASE + "/",
}

DETAIL_RE = re.compile(r"^/huayao/hy\d+[a-z]?\.htm$", re.IGNORECASE)
LETTER_RE = re.compile(r"^/huayao1/[a-z]\d*\.htm$", re.IGNORECASE)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "drug_data.jsonl")
SEEN_PATH = os.path.join(DATA_DIR, "seen_urls.txt")


def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s


SESSION = build_session()


def get_html(url: str) -> str:
    try:
        r = SESSION.get(url, timeout=15)
        if r.status_code != 200:
            print(f"[HTTP {r.status_code}] {url}")
            return ""
        # 该站编码不稳定，统一用 apparent_encoding 兜底
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except Exception as e:
        print(f"[ERR] {url} -> {e}")
        return ""


def get_letter_pages() -> list[str]:
    """从入口提取字母索引页 URL 列表。"""
    html = get_html(ENTRY)
    if not html:
        return []
    tree = etree.HTML(html)
    hrefs = tree.xpath('//a/@href')
    letters = []
    seen = set()
    for h in hrefs:
        if h and LETTER_RE.match(h) and h not in seen:
            seen.add(h)
            letters.append(urljoin(BASE, h))
    return letters


def parse_letter(url: str) -> list[str]:
    """从字母索引页提取所有详情页 URL。"""
    html = get_html(url)
    if not html:
        return []
    tree = etree.HTML(html)
    hrefs = tree.xpath('//a/@href')
    out = []
    seen = set()
    for h in hrefs:
        if h and DETAIL_RE.match(h) and h not in seen:
            seen.add(h)
            out.append(urljoin(BASE, h))
    return out


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def parse_detail(url: str, debug: bool = False) -> dict | None:
    html = get_html(url)
    if not html:
        return None

    tree = etree.HTML(html)

    name = tree.xpath('string(//h1[contains(@class,"yaopinming")])').strip()
    if not name:
        # 退化：从 title 抠
        title = tree.xpath('string(//title)').strip()
        name = title.split("_")[0] if title else ""

    items = tree.xpath('//li[contains(@class,"smsli")]')
    if not items:
        if debug:
            print(f"[skip] 无 smsli 项: {url}")
        return None

    data: dict = {"药品名称": name} if name else {}

    field_pat = re.compile(r"^【\s*(.+?)\s*】(.*)$", re.DOTALL)

    for li in items:
        # 用 \n 拼接 <br> 后再 strip
        for br in li.xpath('.//br'):
            br.tail = ("\n" + br.tail) if br.tail else "\n"
        raw = li.xpath('string(.)')
        raw = _clean(raw)
        m = field_pat.match(raw)
        if not m:
            continue
        field = m.group(1).strip()
        value = m.group(2).strip()
        if not field:
            continue

        # 【药品名称】 内部是多行 "子字段：值"，拆开作为独立字段
        if field == "药品名称":
            for line in value.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "：" in line:
                    k, v = line.split("：", 1)
                    data[k.strip()] = v.strip()
                elif ":" in line:
                    k, v = line.split(":", 1)
                    data[k.strip()] = v.strip()
            continue

        # 同字段重复时合并
        if field in data and data[field] != value:
            data[field] = data[field] + "\n" + value
        else:
            data[field] = value

    if len(data) <= 1:
        if debug:
            print(f"[skip] 字段过少: {url}")
        return None

    # 厂家信息（可选，结构化为列表）
    factories = []
    for div in tree.xpath('//div[contains(@class,"changjia_content")]//div[contains(@class,"chanpin")]'):
        spec = div.xpath('string(.//span[contains(@class,"guige")])').strip()
        firm = div.xpath('string(.//a[contains(@class,"qiye")])').strip()
        if spec or firm:
            factories.append({"规格": spec, "生产企业": firm})
    if factories:
        data["生产厂家"] = factories

    data["source_url"] = url
    return data


def load_seen() -> set[str]:
    if not os.path.exists(SEEN_PATH):
        return set()
    with open(SEEN_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_seen(url: str) -> None:
    with open(SEEN_PATH, "a", encoding="utf-8") as f:
        f.write(url + "\n")


def append_record(rec: dict) -> None:
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def crawl(max_count: int = 3000, debug: bool = False) -> int:
    seen = load_seen()
    print(f"已存 source_url: {len(seen)} 条，目标新增至总计 {max_count} 条")

    # 已存就计入总数
    saved = 0
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            saved = sum(1 for _ in f)
    print(f"已存说明书记录: {saved} 条")

    if saved >= max_count:
        print("已达目标，无需再抓。")
        return saved

    letters = get_letter_pages()
    if not letters:
        print("无法获取字母索引页，终止。")
        return saved
    print(f"字母页数: {len(letters)} -> {[u.rsplit('/',1)[-1] for u in letters]}")

    for li_url in letters:
        print(f"\n==== 抓取字母页 {li_url} ====")
        details = parse_letter(li_url)
        print(f"  详情链接: {len(details)}")
        time.sleep(random.uniform(0.6, 1.2))

        for d_url in details:
            if d_url in seen:
                continue
            seen.add(d_url)
            append_seen(d_url)

            rec = parse_detail(d_url, debug=debug)
            time.sleep(random.uniform(0.6, 1.2))

            if not rec:
                continue

            append_record(rec)
            saved += 1
            print(f"  [{saved}/{max_count}] {rec.get('药品名称', '?')}")

            if saved >= max_count:
                print("\n达到目标数量。")
                return saved

    return saved


if __name__ == "__main__":
    print("开始抓取...")
    total = crawl(max_count=3000, debug=False)
    print(f"\n抓取完成，已保存 {total} 条到: {OUTPUT_PATH}")
