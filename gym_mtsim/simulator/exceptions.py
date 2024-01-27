class SymbolNotFound(Exception):
    """
    Raised when a symbol is not found in the symbol table
    """

    pass


class OrderNotFound(Exception):
    """
    Raised when an order is not found in the order book
    """

    pass
