

import os

os.environ['HUGGINGFACEHUB_API_TOKEN']='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'

"""# Load the Open Source Model"""

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256
  )

local_llm = HuggingFacePipeline(pipeline=pipeline)

"""# Language Expression Language (LCEL)"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_1 = ChatPromptTemplate.from_template("How to prepare {topic}")
model_1= local_llm
output_parser_1 = StrOutputParser()

chain_1 = prompt_1 | model_1 | output_parser_1

chain_1.invoke({"topic": "Noodles"})

"""# Prompt Templates

Prompt Template
"""

# PromptTemplate

from langchain.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_template.format(adjective="funny", content="chickens")

"""Chat Prompt Template"""

# ChatPromptTemplate
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

from langchain.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template_string)

#print the formated promt
prompt_template.format(style="Italic", text="what is your name?")

"""Few-Shot Prompt Template"""

#Few shot prompt template

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]

# create a formatter
example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

print(example_prompt.format(**examples[0]))

#feed the question
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(prompt.format(input="Who was the father of Mary Ball Washington?"))

"""# Example Selectors"""

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

example_selector = LengthBasedExampleSelector(
    # The examples it has available to choose from.
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # The maximum length that the formatted examples should be.
    max_length=25
)

dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

# An example with small input, so it selects all examples.
print(dynamic_prompt.format(adjective="big"))

# An example with long input, so it selects only one example.
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
print(dynamic_prompt.format(adjective=long_string))

"""# Output Parser

CSV Parser
"""

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

#define the parser type
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

#create the template
prompt = PromptTemplate(
    template="List five {subject}.",
    input_variables=["subject"]
)

_input = prompt.format(subject="ice cream flavors")
output = local_llm(_input)
output_parser.parse(output)

"""Custom Structured Parser"""

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

#define the structure
response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(
        name="source",
        description="source used to answer the user's question, should be a website.",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

#define the template
prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

_input = prompt.format(question="What is the capital of France?")
output = local_llm(_input)
output_parser.parse(output)

"""# Document loaders

CSV Loader
"""

from langchain.document_loaders import CSVLoader


loader = CSVLoader(file_path='PASTE-YOUR-FILE-PATH')
data = loader.load()

print(data)

"""File Directory"""



from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('PASTE-YOUR-FILE-PATH')    #loader = DirectoryLoader('../', glob="**/*.md") -> glob parameter to control which files to load

docs = loader.load()

len(docs)

print(docs)

"""PDF Loader"""



from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("PASTE-YOUR-FILE-PATH")
pages = loader.load_and_split()

pages[0]

"""# Text Splitters

Split by Character
"""

data = "PASTE-YOUR-FILE-PATH"

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader(data)
pages = loader.load()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=50,
    chunk_overlap=40,
    length_function=len,
)

texts = text_splitter.split_documents(pages)

texts[0]

"""# Text Embedding

HuggingFace Embeddings
"""

#embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings()

#perform embeddings to the data

embeddings = embedding_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)

len(embeddings), len(embeddings[0])

#perform embedding for the query
embedded_query = embedding_model.embed_query("What was the name mentioned in the conversation?")
embedded_query[:5]

"""# Vector Store"""

!pip install faiss-cpu

from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('PASTE-YOUR-TEXT-FILE-PATH').load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0) #change values
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, HuggingFaceEmbeddings())  #store in faiss cpu

#perform similarity search

query = "What is AI?"
docs = db.similarity_search(query)
print(docs[0].page_content)

"""# Retriever"""



from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('PASTE-YOUR-TEXT-FILE-PATH').load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0) #change values
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, HuggingFaceEmbeddings())  #store in chroma db

#define the retriever
retriever = db.as_retriever()

#retrive contents
docs = retriever.get_relevant_documents("Your Query.")
docs

#Similarity score threshold retrieval
#You can also set a retrieval method that sets a similarity score threshold and only returns documents with a score above that threshold.

retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)

docs = retriever.get_relevant_documents("Your Query.")

docs

#Specifying top k
#you can also specify search kwargs like k to use when doing retrieval.

retriever = db.as_retriever(search_kwargs={"k": 1})

docs = retriever.get_relevant_documents("Your Query.")

print("Length of the documnet : ",len(docs))

docs

"""# Agents"""

from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

#load the required tool
tools = load_tools(["llm-math"], llm=local_llm)

#create the Agent

agent= initialize_agent(
    tools,
    local_llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

#run the agent
agent("What is the 25% of 300?")

"""# Chains

LLMChain
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
chain = LLMChain(llm=local_llm, prompt=prompt)
chain.run(product="ice cream")

"""Simple sequential Chain"""

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=local_llm, prompt=first_prompt)

# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=local_llm, prompt=second_prompt)

#combine all the chain in sequential form

overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )

#run the chain
product = "Perfume"
overall_simple_chain.run(product)

"""# Memory

ConversationBufferMemory
"""

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# define the llm, memory and chain.
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=local_llm,
    memory = memory,
    verbose=True
)

#start the conversation
conversation.predict(input="Hi, my name is Ram")

conversation.predict(input="What is 1+1?")

conversation.predict(input="What is my name?")

#print the conversation

print(memory.buffer)

#load the history in the dictionary

memory.load_memory_variables({})

