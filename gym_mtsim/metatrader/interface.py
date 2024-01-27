from enum import Enum
from datetime import datetime

import numpy as np

try:
    # noinspection PyUnresolvedReferences
    import MetaTrader5 as mt5
    from MetaTrader5 import SymbolInfo as MtSymbolInfo
    MT5_AVAILABLE = True
except ImportError:
    MtSymbolInfo = object
    MT5_AVAILABLE = False


class Timeframe(Enum):
    """
    Timeframe is an enumeration of all possible timeframes
    """

    M1 = 1  # mt5.TIMEFRAME_M1
    M2 = 2  # mt5.TIMEFRAME_M2
    M3 = 3  # mt5.TIMEFRAME_M3
    M4 = 4  # mt5.TIMEFRAME_M4
    M5 = 5  # mt5.TIMEFRAME_M5
    M6 = 6  # mt5.TIMEFRAME_M6
    M10 = 10  # mt5.TIMEFRAME_M10
    M12 = 12  # mt5.TIMEFRAME_M12
    M15 = 15  # mt5.TIMEFRAME_M15
    M20 = 20  # mt5.TIMEFRAME_M20
    M30 = 30  # mt5.TIMEFRAME_M30
    H1 = 1 | 0x4000  # mt5.TIMEFRAME_H1
    H2 = 2 | 0x4000  # mt5.TIMEFRAME_H2
    H4 = 4 | 0x4000  # mt5.TIMEFRAME_H4
    H3 = 3 | 0x4000  # mt5.TIMEFRAME_H3
    H6 = 6 | 0x4000  # mt5.TIMEFRAME_H6
    H8 = 8 | 0x4000  # mt5.TIMEFRAME_H8
    H12 = 12 | 0x4000  # mt5.TIMEFRAME_H12
    D1 = 24 | 0x4000  # mt5.TIMEFRAME_D1
    W1 = 1 | 0x8000  # mt5.TIMEFRAME_W1
    MN1 = 1 | 0xC000  # mt5.TIMEFRAME_MN1


def initialize() -> bool:
    """
    function to initialize MetaTrader5
    :return: bool: True if MetaTrader5 is initialized, False otherwise
    """

    _check_mt5_available()
    return mt5.initialize()


def shutdown() -> None:
    """
    function to shut down MetaTrader5
    :return: None
    """

    _check_mt5_available()
    mt5.shutdown()


def copy_rates_range(symbol: str, timeframe: Timeframe, date_from: datetime, date_to: datetime) -> np.ndarray:
    """
    function to retrieve historical data from MetaTrader5
    :param symbol: str: symbol name
    :param timeframe: Timeframe: timeframe of the data (e.g. possible values: M1, M5, M15, M30, H1, H4, D1, W1, MN1)
    :param date_from: datetime: start date
    :param date_to: datetime: end date
    :return: np.ndarray: historical data
    """

    _check_mt5_available()
    return mt5.copy_rates_range(symbol, timeframe.value, date_from, date_to)


def symbol_info(symbol: str) -> MtSymbolInfo:
    """
    function to retrieve symbol info from MetaTrader5
    :param symbol: str: symbol name
    :return: MtSymbolInfo: symbol info
    """

    _check_mt5_available()
    return mt5.symbol_info(symbol)


def _check_mt5_available() -> None:
    """
    private function to check if MetaTrader5 is available
    :return: None
    """

    if not MT5_AVAILABLE:
        raise OSError("MetaTrader5 is not available on your platform.")
