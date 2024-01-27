from typing import Tuple

import pytz
import calendar
from datetime import datetime
import pandas as pd

from . import interface as mt
from .symbol import SymbolInfo


def retrieve_data(
        symbol: str,
        from_dt: datetime,
        to_dt: datetime,
        timeframe: mt.Timeframe
) -> Tuple[SymbolInfo, pd.DataFrame]:
    """
    function to retrieve historical data from MetaTrader5
    :param symbol: str: symbol name
    :param from_dt: datetime: start date
    :param to_dt: datetime: end date
    :param timeframe: mt.Timeframe: timeframe of the data (e.g. possible values: M1, M5, M15, M30, H1, H4, D1, W1, MN1)
    :return: Tuple[SymbolInfo, pd.DataFrame]: symbol info and historical data
    """

    # check if MetaTrader is initialized
    if not mt.initialize():
        raise ConnectionError(f"MetaTrader cannot be initialized")

    # get the symbol info
    symbol_info = _get_symbol_info(symbol)

    # convert dates to UTC
    utc_from = _local2utc(from_dt)
    utc_to = _local2utc(to_dt)
    all_rates = []

    partial_from = utc_from
    partial_to = _add_months(partial_from, 1)

    # get the data in chunks of 1 month
    while partial_from < utc_to:
        rates = mt.copy_rates_range(symbol, timeframe, partial_from, partial_to)
        all_rates.extend(rates)
        partial_from = _add_months(partial_from, 1)
        partial_to = min(_add_months(partial_to, 1), utc_to)

    all_rates = [list(r) for r in all_rates]

    # convert the data chunks into a single DataFrame
    rates_frame = pd.DataFrame(
        all_rates,
        columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', '_', '_'],
    )
    rates_frame['Time'] = pd.to_datetime(rates_frame['Time'], unit='s', utc=True)

    data = rates_frame[['Time', 'Open', 'Close', 'Low', 'High', 'Volume']].set_index('Time')
    data = data.loc[~data.index.duplicated(keep='first')]

    mt.shutdown()

    return symbol_info, data


def _get_symbol_info(symbol: str) -> SymbolInfo:
    """
    private function to retrieve symbol info from MetaTrader5
    :param symbol: str: symbol name
    :return: SymbolInfo: symbol info
    """

    info = mt.symbol_info(symbol)
    symbol_info = SymbolInfo(info)
    return symbol_info


def _local2utc(dt: datetime) -> datetime:
    """
    private function to convert local datetime to UTC
    :param dt: datetime: local datetime
    :return: datetime: UTC datetime
    """

    return dt.astimezone(pytz.timezone('Etc/UTC'))


def _add_months(sourcedate: datetime, months: int) -> datetime:
    """
    private function to add months to a datetime
    :param sourcedate: datetime: source datetime
    :param months: int: number of months to add
    :return: datetime: new datetime
    """

    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])

    return datetime(
        year, month, day,
        sourcedate.hour, sourcedate.minute, sourcedate.second,
        tzinfo=sourcedate.tzinfo
    )
