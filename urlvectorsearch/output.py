from langchain.llms import OpenAI


def make_description(que_, docu_, llm_):
    model_name="gpt-3.5-turbo"
    output_llm = OpenAI(temperature=0, model_name=model_name)

    prompt = """
    {summary}は{query}についてのどのような記事か100文字程度で教えてください．
    """.format(summary = docu_, query = que_)

    return output_llm(prompt)



def description_output_with_scoer(qdrant_, query_, num_output=5):
    found_docs = qdrant_.similarity_search_with_score(query_)

    output_llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

    output_ = ""
    for i in range(num_output):
        # 文章とスコアの取得
        docu, score = found_docs[i]
        # リンクの取得
        url = found_docs[i][0].metadata['url']

        description = make_description(query_, docu, output_llm)

        output_ += f"{description}" + "\n" + f"score：{score}" + "\n" + f"URL：{url}" + "\n"
    
    return output_


def description_output(qdrant_, query_, num_output=5):
    found_docs = qdrant_.similarity_search(query_)

    output_llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

    output_ = ""
    for i in range(num_output):
        # 文章の取得
        docu = found_docs[i].page_content
        # リンクの取得
        url = found_docs[i].metadata['url']

        description = make_description(query_, docu, output_llm)

        output_ += f"{description}" + "\n" + f"{url}" + "\n"
    
    return output_