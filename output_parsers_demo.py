from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("wire a short poem about {topic}")

# NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
llm = init_chat_model(model="gpt-6-luna", reasoning_effort="none", temperature=0)

chain = prompt | llm | parser

response = chain.invoke({"topic": "nature"})

print(type(response))


# JsonOutputParser example
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with 'name' and 'age' for: {description}"
)

chain = prompt | llm | parser

result = chain.invoke({"description": "A 25-year-old developer named Alex"})
print(result)  # {'name': 'Alex', 'age': 25}

# PydanticOutputParser example
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's occupation")


parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with 'name', 'age', and 'occupation' for: {description}"
).partial(format_instructions=parser.get_format_instructions())
chain = prompt | llm | parser
result = chain.invoke({"description": "A 30-year-old artist named Maria"})
print(result)  # Person(name='Maria', age=30, occupation='artist')


# Structured Output
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")


# Bind the schema to the model
structured_model = llm.with_structured_output(MovieReview)

result = structured_model.invoke("Review: Inception is a mind-bending thriller. 9/10")
print(result)  # MovieReview(title='Inception', review='A mind-bending thriller.', rating=9)