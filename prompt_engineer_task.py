import chromadb
from openai import OpenAI

# 1. 初始化 DeepSeek 客户端
client = OpenAI(
    api_key="...",
    base_url="https://api.deepseek.com"
)

# 2. 连接构建的向量数据库
db_client = chromadb.PersistentClient(path="chroma_db")
collection = db_client.get_collection("drug_instructions")


# 3. 定义检索工具：专门提取安全性相关字段
def search_drug_safety_info(drug_name):
    # 我们仅关注安全性强相关的字段以提高 RAG 信噪比 [cite: 17]
    safety_fields = ["禁忌", "药物相互作用", "注意事项", "不良反应"]

    results = collection.get(
        where={
            "$and": [
                {"drug_name": drug_name},
                {"field": {"$in": safety_fields}}
            ]
        }
    )

    if not results['documents']:
        return f"找不到 {drug_name} 的相关说明。"

    return "\n".join(results['documents'])


# 4. Prompt 工程师的核心任务：设计 ReAct 系统提示词
SYSTEM_PROMPT = """
你是一位极其严谨的临床药师。你的任务是判断用户提供的药物组合是否存在配伍禁忌。

# 协议 (ReAct)
1. **Thought**: 思考需要查询哪种药品的说明书。
2. **Action**: 提取说明书中关于【禁忌】和【相互作用】的原文。
3. **Observation**: 仔细阅读原文。
4. **Final Answer**: 基于原文给出结论。如果原文没写有冲突，就说“未见明确记录”，严禁幻觉。

# 约束
- 必须基于提供的说明书片段回答。
- 必须指出具体哪个字段提到了冲突。
"""


# 5. 执行检查任务
def check_interaction(drug_list):
    # 第一步：检索所有相关药物的上下文
    context = ""
    for drug in drug_list:
        context += f"\n【{drug} 说明书摘录】：\n{search_drug_safety_info(drug)}\n"

    # 第二步：调用 DeepSeek 进行推理
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"上下文信息：{context}\n\n问题：请问 {' 和 '.join(drug_list)} 能一起吃吗？"}
        ],
        stream=False
    )

    print("\n--- DeepSeek 药师推理结果 ---")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    # 测试：尝试检查提到的常见配伍冲突（如阿司匹林）
    check_interaction(["阿司匹林肠溶片", "华法林钠片"])