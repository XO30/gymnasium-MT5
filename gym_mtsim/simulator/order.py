from enum import IntEnum
from datetime import datetime


class OrderType(IntEnum):
    """
    OrderType is a representation of the type of order. It can be either a Buy or a Sell.
    """

    Sell = 0
    Buy = 1

    @property
    def sign(self) -> float:
        """
        property method to return the sign of the order type.
        :return: float: 1 if the order is a buy, -1 if the order is a sell.
        """

        return 1. if self == OrderType.Buy else -1.

    @property
    def opposite(self) -> 'OrderType':
        """
        property method to return the opposite order type.
        :return: OrderType: the opposite order type.
        """
        if self == OrderType.Sell:
            return OrderType.Buy
        return OrderType.Sell


class Order:
    """
    Order is a representation of a trade. It contains all the information about the trade.

    :param id: int: the id of the order.
    :param order_type: OrderType: the type of the order.
    :param symbol: str: the symbol of the order.
    :param volume: float: the volume of the order.
    :param fee: float: the fee of the order.
    :param entry_time: datetime: the entry time of the order.
    :param entry_price: float: the entry price of the order.
    :param exit_time: datetime: the exit time of the order.
    :param exit_price: float: the exit price of the order.

    :ivar id: int: the id of the order.
    :ivar type: OrderType: the type of the order.
    :ivar symbol: str: the symbol of the order.
    :ivar volume: float: the volume of the order.
    :ivar fee: float: the fee of the order.
    :ivar entry_time: datetime: the entry time of the order.
    :ivar entry_price: float: the entry price of the order.
    :ivar exit_time: datetime: the exit time of the order.
    :ivar exit_price: float: the exit price of the order.
    :ivar profit: float: the profit of the order.
    :ivar margin: float: the margin of the order.
    :ivar closed: bool: whether the order is closed or not.
    """

    def __init__(
        self,
        id: int, order_type: OrderType,
        symbol: str,
        volume: float,
        fee: float,
        entry_time: datetime,
        entry_price: float,
        exit_time: datetime,
        exit_price: float
    ) -> None:

        self.id = id
        self.type = order_type
        self.symbol = symbol
        self.volume = volume
        self.fee = fee
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.profit = 0.
        self.margin = 0.
        self.closed = False
