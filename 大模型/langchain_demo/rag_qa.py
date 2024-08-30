from service.retrievers import VectorRetriever


class LlmRagQa:
    """基于召回的智能问答"""
    def __init__(self, name="ragqa", score=0.75, policy=1, quote_tip="引用知识", system_prompt=None, llm_model=None, embed_model_path=None, init_kn_files=[]):
        self.name = name
        self.model = llm_model
        self.policy = policy
        self.score = score
        self.quote_tip = quote_tip
        self.system_prompt = system_prompt
        self.init_kn_files = init_kn_files
        self.retriever = VectorRetriever(self.name, embed_model_path=embed_model_path)
        self.load_init_file()

    def load_init_file(self):
        if not self.retriever.is_load_from_persist:
            for loader in self.init_kn_files:
                for keys, metas, citations in loader.gen_embed_keys_citation():
                    self.retriever.add_keys(keys, metas, citations)

    def stream_chat(self, query, history=[], need_accord=False):
        accord_list = self.retriever.retrieve(query, top_n=3, score=self.score, policy=self.policy)
        print("accord_list:", accord_list)
        if not accord_list and need_accord:
            yield "无相关知识，暂时还回答不了你的问题，我还在学习中……", history
        else:
            systemc_message = self.system_prompt.format(accord="\n\n".join(map(lambda x: x.page_content, accord_list)))
            chat_hist = [{"role": "system", "content": systemc_message}]
            rep, his = None, None
            for rep, his in self.model.stream_chat(query, chat_hist):
                if len(rep) < 4:
                    continue
                if rep[0:4] == "无法回答":
                    yield "无相关知识，暂时还回答不了你的问题，我还在学习中……", history
                    break
                else:
                    yield rep, his
            his.append({"role":"assistant", "content":rep})
            if accord_list:
                citation = "\n*{}:*\n".format(self.quote_tip)
                for doc in accord_list:
                    citation += "*{}: {}*\n".format(doc.metadata.get("type"), doc.metadata.get("title"))
                yield rep+"<br><br>\n***" + citation, his

    def add_file(self):
        pass

    def del_file(self):
        pass

    def list_file(self):
        pass

