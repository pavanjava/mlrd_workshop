from pydantic import BaseModel
from typing import List


class Body(BaseModel):
    query: str
    model: str
    include_tables: List[str] | None
