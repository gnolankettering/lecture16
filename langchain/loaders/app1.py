from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import config

llm = OpenAI(temperature=0, api_key=config.OPENAI_API_KEY)

# 1. Document Loading
loader = DirectoryLoader('./data', glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# 2. Text Splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(documents)

# 3. Summarization Chain
summarize_template = PromptTemplate.from_template(
    "Summarize the following text in 2-3 sentences:\n\n{text}"
)
summarize_chain = summarize_template | llm | StrOutputParser()

# 4. Question Answering Chain
qa_template = PromptTemplate.from_template(
    "Based on the following summaries, answer the question.\n\n"
    "Summaries: {summaries}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
qa_chain = (
    RunnablePassthrough.assign(
        summaries=lambda x: "\n".join(summarize_chain.batch(x["splits"]))
    )
    | qa_template
    | llm
    | StrOutputParser()
)

# 5. Main Process
question = "What are the main topics discussed in these documents?"
result = qa_chain.invoke({"splits": splits, "question": question})

print("Summaries:")
for summary in summarize_chain.batch(splits):
    print(summary)
    print("---")

print("\nQuestion:", question)
print("Answer:", result)