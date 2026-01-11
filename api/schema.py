from pydantic import BaseModel,Field
from typing import List,Optional,Tuple
from datetime import datetime
from typing_extensions import Annotated

MAX_HISTORY=60

class PredictionRequest(BaseModel):
    """
    Input schema
    """
    recent_prices: Annotated[
        List[float],
        Field(
            min_items=10,
            max_items=MAX_HISTORY,
            description=f"Recent closing prices (max {MAX_HISTORY}, most recent last)"
        )
    ]

    date: Optional[str] = Field(
        None,
        description="Optional reference date (YYYY-MM-DD)"
    )


class PredictionOutcome(BaseModel):
    prediction:float
    model_type:str
    arima_order:Tuple[int,int,int]
    timestamp:datetime