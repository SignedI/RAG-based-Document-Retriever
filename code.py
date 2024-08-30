python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install python-dotenv langchain langchain-openai openai chroma unstructured tiktoken lark

OPENAI_API_KEY = ""

import os
from dotenv import load_dotenv
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma

load_dotenv()

class RAG():
    def __init__(self,
                 docs_dir: str,
                 n_retrievals: int = 4,
                 chat_max_tokens: int = 3097,
                 model_name = "gpt-3.5-turbo",
                 creativeness: float = 0.7):
        self.__model = self.__set_llm_model(model_name, creativeness)
        self.__docs_list = self.__get_docs_list(docs_dir)
        self.__retriever = self.__set_retriever(k=n_retrievals)
        self.__chat_history = self.__set_chat_history(max_token_limit=chat_max_tokens)

    def __set_llm_model(self, model_name = "gpt-3.5-turbo", temperature: float = 0.7):
        return ChatOpenAI(model_name=model_name, temperature=temperature)

    def __get_docs_list(self, docs_dir: str) -> list:
        print("Loading documents...")
        loader = DirectoryLoader(docs_dir,
                                 recursive=True,
                                 show_progress=True,
                                 use_multithreading=True,
                                 max_concurrency=4)
        docs_list = loader.load_and_split()
        return docs_list

    def __set_retriever(self, k: int = 4):
        # Chroma Vector Store
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(
            self.__docs_list,
            embedding=embeddings,
            collection_name="personal_documents",
            persist_directory=os.getenv("CHROMA_PERSIST_DIR"),  # Assuming persistence is needed
        )

        # Self-Querying Retriever
        metadata_field_info = [
            AttributeInfo(
                name="source",
                description="The directory path where the document is located",
                type="string",
            ),
        ]

        document_content_description = "Personal documents"

        _retriever = SelfQueryRetriever.from_llm(
            self.__model,
            vector_store,
            document_content_description,
            metadata_field_info,
            search_kwargs={"k": k}
        )

        return _retriever

    def __set_chat_history(self, max_token_limit: int = 3097):
        return ConversationTokenBufferMemory(llm=self.__model, max_token_limit=max_token_limit, return_messages=True)

    def ask(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant responsible for answering questions about documents. Answer the user's question with a reasonable level of detail based on the following context document(s):\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
       
        output_parser = StrOutputParser()
        chain = prompt | self.__model | output_parser
        answer = chain.invoke({
            "input": question,
            "chat_history": self.__chat_history.load_memory_variables({})['history'],
            "context": self.__retriever.get_relevant_documents(question)
        })

        # Update the chat history
        self.__chat_history.save_context({"input": question}, {"output": answer})
       
        return answer

# Running the chatbot
rag = RAG(
    docs_dir='docs', # Directory where documents are located
    n_retrievals=1, # Number of documents retrieved by the search (int)  :   default=4
    chat_max_tokens=3097, # Maximum number of tokens that can be used in chat memory (int)  :   default=3097
    creativeness=1.2, # How creative the response will be (float 0-2)  :   default=0.7
)

print("\nType 'exit' to exit the program.")
while True:
    question = input("Question: ")
    if question == "exit":
        break
    answer = rag.ask(question)
    print('Answer:', answer)
