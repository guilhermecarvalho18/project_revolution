# %%
# from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers import PydanticOutputParser

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field, validator
from typing import Union, List, Dict, TypedDict

import json
import openai
import os

# %%
llm = ChatOpenAI(
    model_name="gpt-4-1106-preview",
    temperature=0.8,
    # 0 = + rígido
    # 1 = - rígido
)

# %%

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NUMBEO_API_KEY = os.getenv("NUMBEO_API_KEY")

#NUMBEO_API_KEY = "8ljlas8k0hxkvw"
#OPENAI_API_KEY = "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"
# %%
#  TEMPLATE
PERSONA_TEMPLATE = """As an immigration consultant, you're assisting clients to find for the most suitable destination city to immigrate to Canada based on the provided information. Ensure responses are polite, and if fitting locations are not known, reply with "Sorry, can you refine the entry?".

Instructions:
{context}

Question - assume that the question is about a client:
{question}

Give [Rate] from 1 to 5 for each item, for each of three comparable cities, and summarize them
Must follow the format and instructions below and don't do different of this format:
[Rate] receives [1/5]
{format_instructions}
Each city in JSON must be a new list inside the string.
Number each city name as key: "city 1 to 3", value: "name".
At the end of the summary, add the comparison of the three cities and tell the sources, with names and links:
comparison: str = Field(
    description="Comparison of the three cities",
    examples="[Feel free to compare all of the cities above]",
)
Tell the sources, with names and links:
source: str = Field(
    description="Source of the data",
    examples="[Describe the sources of the data, like where you found the cost of living, the wage, the education etc. with names and links]",
)
"""

# %%
CONTEXT = """1. The main focus is on the client's career and salary,
searching for the best place to work.
Bring the highest annual salary versus cost of living, then how it fits
into the overall outlook of the chosen city.

2. Begin with the most critical information for the client's quality of life,
mostly based in the balance between annual salary and cost of living.

3. If the spouse's profession is not provided, consideer that I'm single.

4. If the spouse's profession is provided, make a plan for me and a plan for him/her (career and wage).

5. If the number of children is 0, consider that I don't have children.

6. If the salary is not provided, try looking for the highest salary
for the career to find the best city to immigrate.

7. If you don't know some information, don't say that you don't know,
try to infer it and just show the result.

8. Provide options for ONE small city, ONE medium city and ONE large city,
each with its name, climate, and cost of living.
Comment the outlook of each topic, for each three cities.

9. Each city must belong to different provinces.

10. Include additional factors for each city, such as healthcare, education,
safety and cultural insights.

11. If spouse is provided, in each topic, make one plan for me and one plan for the spouse.

12. Offer a brief comparison between the THREE cities to assist the client
in making an informed decision.

13. Add an overview of each city that suits the client's needs, 
explaining why it was chosen, including comments mainly on career, wage,
cost of living, then comment about climate,
and other relevant factors.

14. Enumerate the cities in the order of the best to the worst.

15. Be free to search from different reliable sources.

16. Describe the sources, with links to the used sources.
"""


# %%
class PredictInputSchema(BaseModel):
    occupation: str
    networth: Union[int, str]
    nationality: str
    spouse_occupation: str = None
    number_of_children: int = 0
    activities: str = None


class PredictOutputSchema(BaseModel):
    cities: Dict[str, str] = Field(
        ...,
        description="Dictionary of cities with their names and descriptions.",
    )
    city: Dict[str, dict] = Field(
        ...,
        description="Name of the city.",
        examples="City Name, Province Name",
    )
    population: str = Field(
        ...,
        description="Number of inhabitants.",
        examples="1,000,000",
    )
    profession: str = Field(
        ...,
        description="Profession information and outlook for careers in the city for me and for my spouse. Separate in two sentences: one for me and one for the spouse (begin with a rate 1 to 5).",
        examples="[Rate] - Dentist / [Rate] - Scientist",
    )
    wage: str = Field(
        ...,
        description="Wage information for both me and my spouse (begin with a rate 1 to 5).",
        examples="[Rate] CAD 120,000 / [Rate] CAD 80,000",
    )
    career: str = Field(
        ...,
        description="Career information for both me and my spouse (begin with a rate 1 to 5).",
        # examples="[Rate] CAD 120,000 / [Rate] CAD 80,000",
    )
    climate: str = Field(
        ...,
        description="Climate with example of the course of the year (begin with a rate 1 to 5).",
        examples="[Rate] - Mild winters and warm summers, with temperatures ranging from -3°C to 28°C.",
    )
    cost_of_living: str = Field(
        ...,
        description="Cost of living information with a rating and examples. Be creative on this and bring diverse options, cheap and expensive items with its price (begin with a rate 1 to 5).",
        examples="[Rate] - CAD 36,000 yearly for a family of four. The price [item] is CAD 1, a [item] is CAD 40,000, the rent is CAD 1,000 and the cost of [item] is CAD 1.",
    )
    healthcare: str = Field(
        ...,
        description="Healthcare quality rating and comments (begin with a rate 1 to 5).",
        examples="[Rate] - High-quality healthcare services with several clinics and a major hospital.",
    )
    education: str = Field(
        ...,
        description="Education system rating and comments, bring some reputables schools and universities informations.  Separate in two sentences: one for me and one for the spouse (begin with a rate 1 to 5).",
        examples="[Rate] - Good public and private schools and kindergartens for the kids, and the university [name of the university] campus offering a variety of programs [list of programs if possible that suits to me and my wife's career].",
    )
    safety: str = Field(
        ...,
        description="Comments about the crime scene in the city (begin with a rate 1 to 5).",
        examples="[Rate] - [Comment about the crime scene in the city and the why, if it's possible].",
    )
    cultural_insights: str = Field(
        ...,
        description="Cultural insights rating, information, and comments that fits to my hobbies and interests (begin with a rate 1 to 5).",
        examples="[Rate] - Growing diversity, but limited to [comunity].",
    )
    overview: str = Field(
        ...,
        description="Brief overview of the city with a rating. Consider my hobbies and interests. (start with a rating from 1 to 5).",
        examples="4/5 - [City name] is selected for its favorable climate, solid wage prospects for [profession], and overall high quality of life.",
    )


# %%
def predict(input_data: PredictInputSchema, PredictOutputSchema):
    question = f"""I am a professional in {input_data.occupation}, \
    with a net worth of {input_data.networth}, planning to relocate to Canada with or without my spouse, \
    a {input_data.spouse_occupation}, and our {input_data.number_of_children} children. \
    We seek a city or town that offers high earnings potential, low cost of living, \
    and high quality of life. My nationality is {input_data.nationality} and \
    I enjoy {input_data.activities}."""

    parser = PydanticOutputParser(pydantic_object=PredictOutputSchema)
    format_instructions = parser.get_format_instructions()

    # persona_prompt = PromptTemplate.from_template(template=PERSONA_TEMPLATE)
    persona_prompt = PromptTemplate.from_template(
        template=PERSONA_TEMPLATE,
        partial_variables={"format_instructions": format_instructions},
    )
    runnable = persona_prompt | llm | StrOutputParser()
    return runnable.invoke({"context": CONTEXT, "question": question})


# %%
# input_data = PredictInputSchema(
#     occupation="carpinteer",
#     networth="20 per hour",
#     nationality="Brazilian",
#     spouse_occupation="",
#     number_of_children=0,
#     activities="hiking, soccer and voleyball",
# )

# %%
# output = predict(input_data, PredictOutputSchema)
# %%
# print(output)
# %%
# json_output = json.loads(output[7:-3])
# print(json_output)


# %%
if __name__ == "__main__":
    input_data = PredictInputSchema(
        occupation="CEO of an early stage Fintech Startup",
        networth=120000,
        nationality="Brazil",
        spouse_occupation="Marketing Director",
        number_of_children=1,
        activities="Spend the life in expansive resorts and also making trekkings in the nature. Appreciate World sophisticated cousine, rare top cigars and wines. Bring cities from the same province.",
    )
    predict_output = json.loads(predict(input_data, PredictOutputSchema)[7:-3])
    predict = print(predict_output)