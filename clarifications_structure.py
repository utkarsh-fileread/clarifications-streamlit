import pydantic
from typing import Union, Literal


class FollowUps(pydantic.BaseModel):
    more_follow_ups: list[str] = pydantic.Field(
        default=...,
        description="More follow-up questions based on the user's responses. Tailored to the context that can help get us getting all necessary information to answer the question to the best of our ability",
    )
    summary: str = pydantic.Field(
        default=...,
        description="A summary of questions that were asked and answered. A line of summary explaining line of questioning.",
    )

class NeedMoreClarifications(pydantic.BaseModel):
    need_more: Union[Literal[False], FollowUps] = pydantic.Field(
        default=...,
        description="Your job is to identify areas where more information is needed and ask for it.",
    )