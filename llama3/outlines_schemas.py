from pydantic import BaseModel
from typing import List

class ExpectedJSONOutputFormat_Dates(BaseModel):
    dates: List[str] = None