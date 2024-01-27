from typing import Tuple

from .interface import MtSymbolInfo


class SymbolInfo:
    """
    SymbolInfo is a wrapper around MtSymbolInfo that provides a more convenient interface.

    :param info: MtSymbolInfo: symbol info

    :ivar name: str: the name of the symbol.
    :ivar market: str: the market of the symbol.
    :ivar currency_margin: str: the margin currency of the symbol.
    :ivar currency_profit: str: the profit currency of the symbol.
    :ivar currencies: Tuple[str, ...]: the currencies of the symbol.
    :ivar trade_contract_size: float: the contract size of the symbol.
    :ivar margin_rate: float: the margin rate of the symbol.
    :ivar volume_min: float: the minimum volume of the symbol.
    :ivar volume_max: float: the maximum volume of the symbol.
    :ivar volume_step: float: the volume step of the symbol.
    """
    def __init__(self, info: MtSymbolInfo) -> None:
        self.name: str = info.name
        self.market: str = self._get_market(info)

        self.currency_margin: str = info.currency_margin
        self.currency_profit: str = info.currency_profit
        self.currencies: Tuple[str, ...] = tuple({self.currency_margin, self.currency_profit})

        self.trade_contract_size: float = info.trade_contract_size
        self.margin_rate: float = 1.0  # MetaTrader info does not contain this value!

        self.volume_min: float = info.volume_min
        self.volume_max: float = info.volume_max
        self.volume_step: float = info.volume_step


    def __str__(self) -> str:
        return f'{self.market}/{self.name}'

    @staticmethod
    def _get_market(info: MtSymbolInfo) -> str:
        """
        private static method to get the market from the path of the symbol
        :param info: MtSymbolInfo: symbol info
        :return: str: market name
        """

        mapping = {
            'forex': 'Forex',
            'crypto': 'Crypto',
            'stock': 'Stock',
        }

        root = info.path.split('\\')[0]
        for k, v in mapping.items():
            if root.lower().startswith(k):
                return v

        return root
