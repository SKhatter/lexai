import streamlit as st 
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI as langchainOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback 
import requests
import re
from langchain.retrievers.you import YouRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from openai import OpenAI

CLEANR = re.compile('<.*?>')

# Sidebar contents
with st.sidebar:
    st.title("About Us")
    st.markdown('''
        ## 
            AI e-discovery and 
            drafting tool to 
            simplify search 
            from our database
            of laws.
            
            This is LLM powered
            app, powered using 
            You APIs, Langchain,
            OpenAI models.                
        ''')
    add_vertical_space(15)
    
    st.write("Authors: Tim, M.Buleandra, Sumedha")


def get_ai_snippets_for_query(query):
    headers = {"X-API-Key": 'bc9baa56-7f46-44b5-872f-07f2e37de326<__>1ODuqPETU8N2v5f4ZKcMJWpu'}
    params = {"query": query}
    results= requests.get(
        f"https://api.ydc-index.io/search?query={query}",
        params=params,
        headers=headers,
    ).json()
    return results


def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext


def generate_answer(query):
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
        api_key= os.getenv("OPENAI_API_KEY")
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model="gpt-3.5-turbo-16k",
    )
    return chat_completion.choices[0].message.content


def main():
    load_dotenv()
    st.title("Lex.AI")
    query_raw = st.text_area("Type your Question here")
    report = []
    if query_raw:
        query = "justia " + query_raw
        results = get_ai_snippets_for_query(query)
        for hit in results["hits"]:
            sub_report = ""
            if "justia" in hit['url']:
                sub_report = sub_report + "\n" + "**** TITLE: " + cleanhtml(hit['title']) + "  ******" +"\n"
                sub_report = sub_report + "\n" + "CITATION: " + hit['url'] + "\n"
                sub_report = sub_report + "\n"
                for snippet in hit['snippets']:
                    sub_report +=  cleanhtml(snippet)
                #report = report + "\n"+ cleanhtml(hit['description']) + cleanhtml(hit['snippets'][0])
                sub_report= sub_report + "\n"
                report.append(sub_report)
        st.write('\n----------------------\n'.join(report))


    if report:
        st.subheader('Generating the answer....')
        final_prompt = "use this text as context: " + ' '.join(report) + \
                "answer this questions "+ query_raw +"and mention the citation of the answer you provide"
        response_gpt = generate_answer(final_prompt)
        st.write(response_gpt)

        
        st.subheader('Generating Embeddings.....')
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size= 200,
                chunk_overlap=30, #chunk overlap is important
                length_function=len
            )
        chunks = text_splitter.split_text(text=''.join(report))
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks,embeddings)
        
        st.subheader('...Completed.')

        st.subheader("Ask Questions about the above report....")
        query = st.text_input("Type your Question here")

        if query:
            #print('Hey you', type(vectorstore), list(vectorstore.keys()), list(vectorstore.values()))
            docs = vectorstore.similarity_search(query=query, k=3) #context window, the number of relevant k
            llm = langchainOpenAI(temperature=0)
            #llm = OpenAI(model_name = 'gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:

                response = chain.run(input_documents=docs, question=query)
                print(cb)
                st.write(response)

if __name__ == '__main__':
    main()