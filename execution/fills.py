from dataclasses import dataclass


@dataclass
class Fill:

    order_id: str
    timestamp: str

    side: str

    price: float
    quantity: float

    fee: float
    slippage: float