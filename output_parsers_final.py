"""
Output Parsers and Structured Output in LangChain V.1
"""

from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
model = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)


def demo_str_parser():
    """Basic string output parser."""

    prompt = ChatPromptTemplate.from_template(
        "Give me a one-word answer: What color is the sky?"
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({})
    print(f"Result: '{result}' (type: {type(result).__name__})")


def demo_json_parser():
    """JSON output parser."""

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'city' and 'country' for: {place}\n"
        "Return ONLY valid JSON, no explanation."
    )
    parser = JsonOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"place": "The Eiffel Tower"})
    print(f"Result: {result}")
    print(f"City: {result['city']}, Country: {result['country']}")


def demo_pydantic_parser():
    """Pydantic output parser for type-safe structured data."""

    # Define schema
    class Recipe(BaseModel):
        name: str = Field(description="Name of the recipe")
        ingredients: List[str] = Field(description="List of ingredients")
        prep_time_minutes: int = Field(description="Preparation time in minutes")
        difficulty: str = Field(description="easy, medium, or hard")

    parser = PydanticOutputParser(pydantic_object=Recipe)

    prompt = ChatPromptTemplate.from_template(
        "Create a simple recipe for: {dish}\n\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    result = chain.invoke({"dish": "scrambled eggs"})
    print(f"Recipe: {result.name}")
    print(f"Ingredients: {result.ingredients}")
    print(f"Prep time: {result.prep_time_minutes} mins")
    print(f"Difficulty: {result.difficulty}")

    # Type-safe access
    print(
        f"\nType check - prep_time is int: {isinstance(result.prep_time_minutes, int)}"
    )


def demo_structured_output():
    """Modern with_structured_output() method."""

    class TaskExtraction(BaseModel):
        """Extracted task information."""

        task: str = Field(description="The main task to do")
        priority: str = Field(description="high, medium, or low")
        deadline: Optional[str] = Field(description="Deadline if mentioned")
        assignee: Optional[str] = Field(description="Person assigned if mentioned")

    # Bind schema to model
    structured_model = model.with_structured_output(TaskExtraction)

    # No need for format instructions - it's automatic
    prompt = ChatPromptTemplate.from_template("Extract task information from: {text}")

    chain = prompt | structured_model

    texts = [
        "John needs to finish the report by Friday - it's urgent",
        "We should update the docs sometime next week",
        "Critical: Fix the login bug ASAP",
    ]

    print("Task Extractions:")
    for text in texts:
        result = chain.invoke({"text": text})
        print(f"\nInput: {text}")
        print(f"  Task: {result.task}")
        print(f"  Priority: {result.priority}")
        print(f"  Deadline: {result.deadline}")
        print(f"  Assignee: {result.assignee}")


def demo_complex_schema():
    """Complex nested schema with structured output."""

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class Company(BaseModel):
        name: str
        industry: str
        employee_count: int
        headquarters: Address
        products: List[str]

    structured_model = model.with_structured_output(Company)

    prompt = ChatPromptTemplate.from_template(
        "Extract company information from: {text}"
    )

    chain = prompt | structured_model

    result = chain.invoke(
        {
            "text": "Apple Inc. is a tech company with 160,000 employees based in "
            "Cupertino, California, USA. They make iPhones, MacBooks, and iPads."
        }
    )

    print(f"Company: {result.name}")
    print(f"Industry: {result.industry}")
    print(f"Employees: {result.employee_count}")
    print(f"HQ: {result.headquarters.city}, {result.headquarters.country}")
    print(f"Products: {result.products}")


# Exercise: Extract structured data from text
def exercise_structured_extraction():
    """
    EXERCISE: Create a schema and chain that extracts:
    - Movie title
    - Year released
    - Director
    - Main actors (list)
    - Genre
    - Rating (1-10)

    Test with a movie description.
    """

    class Movie(BaseModel):
        title: str = Field(description="Movie title")
        year: int = Field(description="Year released")
        director: str = Field(description="Director name")
        actors: List[str] = Field(description="Main actors")
        genre: str = Field(description="Primary genre")
        rating: int = Field(description="Rating from 1-10", ge=1, le=10)

    structured_model = model.with_structured_output(Movie)

    prompt = ChatPromptTemplate.from_template(
        "Extract movie information from this review:\n\n{review}"
    )

    chain = prompt | structured_model

    result = chain.invoke(
        {
            "review": "The Dark Knight (2008) directed by Christopher Nolan is an "
            "absolute masterpiece. Christian Bale and Heath Ledger deliver "
            "incredible performances in this action thriller. 10/10!"
        }
    )

    print(f"Title: {result.title}")
    print(f"Year: {result.year}")
    print(f"Director: {result.director}")
    print(f"Actors: {result.actors}")
    print(f"Genre: {result.genre}")
    print(f"Rating: {result.rating}/10")


if __name__ == "__main__":
    print("=" * 50)
    print("Demo 1: String Parser")
    print("=" * 50)
    demo_str_parser()

    print("\n" + "=" * 50)
    print("Demo 2: JSON Parser")
    print("=" * 50)
    demo_json_parser()

    print("\n" + "=" * 50)
    print("Demo 3: Pydantic Parser")
    print("=" * 50)
    demo_pydantic_parser()

    print("\n" + "=" * 50)
    print("Demo 4: Structured Output (Modern)")
    print("=" * 50)
    demo_structured_output()

    print("\n" + "=" * 50)
    print("Demo 5: Complex Schema")
    print("=" * 50)
    demo_complex_schema()

    print("\n" + "=" * 50)
    print("Exercise: Movie Extraction")
    print("=" * 50)
    exercise_structured_extraction()
