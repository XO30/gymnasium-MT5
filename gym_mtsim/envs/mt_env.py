from typing import List, Tuple, Dict, Any, Optional, Union, Callable

import gymnasium

from ..simulator import MtSimulator, OrderType


class MtEnv(gymnasium.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            original_simulator: MtSimulator,
            trading_symbols: List[str],
            window_size: int,
            multiprocessing_processes: Optional[int] = None,
            setting: Dict[str, Any] = None
    ) -> None:

    def _validate_user_input(self, ):
        raise NotImplementedError()

    def reset(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raise NotImplementedError()

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def seed(self, seed: int) -> None:
        raise NotImplementedError()

