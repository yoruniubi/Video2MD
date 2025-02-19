import os
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings, load_index_from_storage, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from openai import OpenAI
from configs import config
class DocumentAnalyzer:
    def __init__(self, documents_path, db_path, db_name, similarity_threshold, chunk_count):
        self.documents_path = documents_path
        self.output_path = db_path
        self.db_name = db_name
        self.similarity_threshold = similarity_threshold
        self.chunk_count = chunk_count
        Settings._embed_model = HuggingFaceEmbedding(model_name="./embedding_model/Jerry0/m3e-base")

    def create_and_save_index(self):
        # 读取文档数据
        documents = SimpleDirectoryReader(self.documents_path).load_data()
        
        # 创建索引（使用VectorStoreIndex并指定嵌入模型）
        index = VectorStoreIndex.from_documents(documents)
        
        # 设置持久化路径
        persist_dir = os.path.join(self.output_path, self.db_name)
        os.makedirs(persist_dir, exist_ok=True)
        
        # 持久化存储
        index.storage_context.persist(persist_dir=persist_dir)

    def get_model_response(self, prompt):
        try:
            # 初始化重排序器
            dashscope_rerank = DashScopeRerank(top_n=self.chunk_count, return_documents=True)
            
            # 加载存储上下文和索引
            storage_context = StorageContext.from_defaults(persist_dir=os.path.join(self.output_path, self.db_name))
            index = load_index_from_storage(storage_context)
            
            # 设置检索器
            retriever = index.as_retriever(similarity_top_k=20)
            retrieved_nodes = retriever.retrieve(prompt)
            print(f"Retrieved {len(retrieved_nodes)} nodes.")
            for idx, node in enumerate(retrieved_nodes):
                print(f"Node {idx + 1}: Score: {node.score}, Text: {node.text[:100]}...")
            
            # 提取节点内容进行重排序
            nodes_to_rerank = [node for node in retrieved_nodes]
            reranked_results = dashscope_rerank.postprocess_nodes(nodes_to_rerank, query_str=prompt)
            
            # 处理检索结果
            chunk_text = []
            for idx, node in enumerate(reranked_results[:self.chunk_count]):
                if hasattr(node, 'score') and node.score >= self.similarity_threshold:
                    chunk_text.append(node.text)
            
            context = "\n".join(chunk_text)
            print(f"Final context passed to AI: {context[:500]}...")
            prompt_template = f"请参考以下内容：{context}\n\n以合适的语气回答用户的问题：{prompt}。如果参考内容中有图片链接也请直接返回。"
        except Exception as e:
            print(f"处理异常: {e}")
            prompt_template = prompt

        # 生成回答
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY") or config['api_key'], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        try:
            completion = client.chat.completions.create(
                model="qwen1.5-0.5b-chat",  
                messages=[{"role": "system", "content": "你是一个专业的文档分析助手"}, {"role": "user", "content": prompt_template}],
                temperature=0.5,
                max_tokens=512
            )
            response = completion.choices[0].message.content
        except Exception as e:
            print(f"生成回答失败: {e}")
            response = "抱歉，暂时无法回答这个问题"
        
        return response

# 示例主函数，用于演示如何初始化和使用DocumentAnalyzer类
if __name__ == '__main__':
    CONFIG = {
        "documents_path": "./transcript",
        "db_path": "./my_db",
        "db_name": "my_database",
        "similarity_threshold": 0.3,
        "chunk_count": 5
    }

    analyzer = DocumentAnalyzer(**CONFIG)

    # 检查索引是否存在
    persist_path = os.path.join(CONFIG["db_path"], CONFIG["db_name"])
    if not os.path.exists(persist_path):
        print("正在创建索引...")
        analyzer.create_and_save_index()

    # 交互循环
    print("\n您好！我是文档分析助手，请输入您的问题（退出请输入 q）")
    while True:
        query = input("\n问题：").strip()
        if query.lower() in ("q", "quit", "exit"):
            break
            
        response = analyzer.get_model_response(prompt=query)
        
        print(f"\n回答：{response}")