import ast
import os

from typing import List
from enum import Enum
from dotenv import load_dotenv

import mlflow
import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from mlflow.entities import SpanType
from pydantic import BaseModel, Field, field_validator


load_dotenv()

# Enable mlflow langchain autologging
mlflow.langchain.autolog()
mlflow.set_experiment("review_analyses")

VALID_CATEGORIES = ["Bad", "Neutral", "Good"]

class Department(str, Enum):
    """Departments in movie Theater"""
    FACILITIES = "facilities"
    FOOD = "food"
    MOVIE_PROGRAM = "movie program"


class Employee(BaseModel):
    """Employee in movie theater"""
    name: str
    department: Department

class Staff(list):
    def __init__(self, employees):
        super().__init__(employees)
        self._lookup = {e.name: e for e in employees}

    def is_valid_employee(self, name: str) -> bool:
        return name in self._lookup

staff = Staff([
    Employee(name="Alice", department=Department.FACILITIES),
    Employee(name="Bob", department=Department.FOOD),
    Employee(name="Chris", department=Department.MOVIE_PROGRAM)
])


class Ticket(BaseModel):
    """Ticket for suggested imporovemnt"""
    employee_name: str = Field(..., description="The name of the employee assigned to the ticket")
    topic: str = Field(..., description="The topic or aspect of the review that needs attention")
    suggested_improvement: str = Field(..., description="The suggested improvement for the topic")
    
    @field_validator("employee_name")
    def validate_employee(cls, v):
        if not staff.is_valid_employee(v):
            raise ValueError(f"Employee '{v}' is not in the staff list.")
        return v

def get_data() -> pd.DataFrame:
    df_complex = pd.read_json("modules/1-Review_Classification/data/complex_reviews.json")
    df_complex.index.name = "review_id"
    return df_complex.reset_index()

def get_llm_client() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_type="azure",
        api_version="2024-12-01-preview",
        azure_endpoint="https://i4talent-openai.openai.azure.com",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )


@mlflow.trace(name="split_revies", span_type=SpanType.TOOL)
def split_review_into_topics(llm, review: str) -> list:
    """
    Uses LLM to split a review into related topics.
    Returns a list of dicts: [{'topic_id': str, 'topic_text': str}]
    """
    prompt_template = PromptTemplate.from_template(
        "Break the following movie review into its related topics or aspects. "
        "Return each topic as a separate string in a Python list. "
        "Do not summarize, just split.\n"
        "Review: {review}"
    )
    prompt = prompt_template.format(review=review)
    response = llm.invoke(prompt)
    topics = ast.literal_eval(response.content)
    return topics

def drop_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.drop(columns=[col])

@mlflow.trace(name="add_topics", span_type=SpanType.CHAIN)
def add_topics_column(df: pd.DataFrame, llm) -> pd.DataFrame:
    """
    Adds a 'Topics' column to the DataFrame by splitting each review into topics using the LLM.
    """
    df['topics'] = df['review'].apply(lambda review: split_review_into_topics(llm, review))
    df = df.explode('topics').reset_index(drop=True)
    return df

@mlflow.trace(name="get_category", span_type=SpanType.TOOL)
def get_category(llm, review: str, valid_categories: List[str]) -> str:
    categories_str = ", ".join(valid_categories)
    prompt_template = PromptTemplate.from_template(
        "Categorize the following movie review as one of the following: {categories_str}. "
        "Only return the category name as a string.\n"
        "Review: {review}"
    )
    prompt = prompt_template.format(categories_str=categories_str, review=review)
    response = llm.invoke(prompt)
    classification = response.content.strip()
    return classification if classification in valid_categories else "Unknown"


def categorize_reviews(df: pd.DataFrame, llm, valid_categories: List[str], review_col: str) -> pd.DataFrame:
    df['Category'] = df[review_col].apply(
        lambda review: get_category(llm, review, valid_categories)
    )
    return df

@mlflow.trace(name="generate_ticket", span_type=SpanType.CHAIN)
def generate_ticket(llm, topic: str, staff: List[Employee]) -> Ticket:
    parser = PydanticOutputParser(pydantic_object=Ticket)
    staff_info = "\n".join([f"- {e.name} ({e.department})" for e in staff])
    format_instructions = parser.get_format_instructions()
    prompt_template = PromptTemplate.from_template(
        "You are assigning a ticket based on the following topic:\n"
        "Topic: {topic}\n\n"
        "The available staff are:\n{staff_info}\n\n"
        "{format_instructions}\n"
        "Choose the most suitable employee and suggest an improvement."
    )
    chain = prompt_template | llm | parser
    response = chain.invoke({
        "topic": topic,
        "staff_info": staff_info,
        "format_instructions": format_instructions
    })
    return response

@mlflow.trace(name="create_tickets", span_type=SpanType.TOOL)
def create_tickets(df: pd.DataFrame, llm, staff: List[Employee]) -> List[Ticket]:
    tickets = []
    for _, row in df.iterrows():
        topic = row['topics']
        ticket = generate_ticket(llm, topic, staff)
        tickets.append(ticket)
    return tickets

@mlflow.trace(name="process_all_reveiws", span_type=SpanType.CHAIN)
def process_reviews_and_generate_tickets(df: pd.DataFrame, llm, staff: List[Employee]) -> pd.DataFrame:
    df_final = (
        df
        .pipe(add_topics_column, llm=llm)
        .pipe(drop_col, "review")
        .pipe(categorize_reviews, llm=llm, valid_categories=VALID_CATEGORIES,  review_col = "topics")
    )
    tickets = create_tickets(df_final, llm=llm, staff=staff)
    return pd.DataFrame([t.model_dump() for t in tickets])




if __name__ == "__main__":
    df_complex = get_data()
    llm = get_llm_client()
    df_results = process_reviews_and_generate_tickets(df_complex, llm, staff)
    print(df_results)