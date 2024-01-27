from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import copy
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from ..simulator import MtSimulator, OrderType
from ta import add_all_ta_features
import pandas as pd



class MtEnvAdvanced(gym.Env):

    metadata = {'render.modes': ['human', 'simple_figure', 'advanced_figure']}

    def __init__(
            self, original_simulator: MtSimulator,
            trading_symbols: List[str],
            window_size: int,
            multiprocessing_processes: Optional[int] = None
    ) -> None:

        # validations
        assert len(original_simulator.symbols_data) > 0, "no data available"
        assert len(original_simulator.symbols_info) > 0, "no data available"
        assert len(trading_symbols) > 0, "no trading symbols provided"
        print('hedging acttivated', original_simulator.hedge)

        for symbol in trading_symbols:
            assert symbol in original_simulator.symbols_info, f"symbol '{symbol}' not found"
            currency_profit = original_simulator.symbols_info[symbol].currency_profit
            assert original_simulator._get_unit_symbol_info(currency_profit) is not None, \
                   f"unit symbol for '{currency_profit}' not found"
        time_points = original_simulator.symbols_data[trading_symbols[0]].index.to_pydatetime().tolist()
        assert len(time_points) > window_size, "not enough time points provided"

        # attributes
        self.seed()
        self.original_simulator = original_simulator
        self.trading_symbols = trading_symbols
        self.window_size = window_size
        self.time_points = time_points
        self.multiprocessing_pool = Pool(multiprocessing_processes) if multiprocessing_processes else None
        self.data_df = NotImplemented

        self.prices = self._get_prices()
        self.signal_features = self._process_data()
        self.features_shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Box(low=-100, high=100, shape=(len(trading_symbols),), dtype=np.float64)

        INF = 1e10
        self.observation_space = spaces.Dict({
            'balance': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'equity': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'margin': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'features': spaces.Box(low=-INF, high=INF, shape=self.features_shape, dtype=np.float64),
            'orders': spaces.Box(low=-INF, high=INF, shape=(len(trading_symbols),), dtype=np.float64)
        })

        # episode
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.time_points) - 1
        self._done: bool = NotImplemented
        self._current_tick: int = NotImplemented
        self.simulator: MtSimulator = NotImplemented
        self.history: dict = NotImplemented


    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self, seed=None, options=None) -> Dict[str, np.ndarray]:
        self._done = False
        self._current_tick = self._start_tick
        self.simulator = copy.deepcopy(self.original_simulator)
        self.simulator.current_time = self.time_points[self._current_tick]
        self.history = dict(
            balance=[self.simulator.balance],
            equity=[self.simulator.equity],
            margin=[self.simulator.margin],
            orders=list(),
        )
        return self._get_observation(), {}


    def step(self, action: np.ndarray) -> any:
        orders = self._apply_action(action)
        if orders is not None:
            self.history['orders'].append(orders)

        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._done = True

        dt = self.time_points[self._current_tick] - self.time_points[self._current_tick - 1]
        self.simulator.tick(dt)

        step_reward = self._calculate_reward()

        self.history['balance'].append(self.simulator.balance)
        self.history['equity'].append(self.simulator.equity)
        self.history['margin'].append(self.simulator.margin)

        observation = self._get_observation()

        if observation['equity'] <= 0:
            self._done = True

        return observation, step_reward, self._done, False, self.history


    def _apply_action(self, action: np.ndarray) -> any:
        orders = {symbol: list() for symbol in self.trading_symbols}
        for i, symbol in enumerate(self.trading_symbols):
            volume = action[i]
            if len(self.simulator.symbol_orders(symbol)) > 0:
                current_volume = self.simulator.symbol_orders(symbol)[0].volume
                order_type = self.simulator.symbol_orders(symbol)[0].type
                if order_type == OrderType.Sell:
                    current_volume = -current_volume
            else:
                current_volume = 0
            volume_to_trade = volume - current_volume
            volume_to_trade = round(volume_to_trade, 2)
            order_type = OrderType.Buy if volume_to_trade > 0 else OrderType.Sell
            try:
                order = self.simulator.create_order(
                    order_type=order_type,
                    symbol=symbol,
                    volume=abs(volume_to_trade),
                    fee=0,
                )
            except:
                order = None
            orders[symbol].append(order)
        return orders


    def _get_prices(self, keys: List[str] = ['Open', 'Close', 'High', 'Low', 'Volume']) -> Dict[str, np.ndarray]:
        prices = {}
        for symbol in self.trading_symbols:
            get_price_at = lambda time: \
                self.original_simulator.price_at(symbol, time)[keys]
            if self.multiprocessing_pool is None:
                p = list(map(get_price_at, self.time_points))
            else:
                p = self.multiprocessing_pool.map(get_price_at, self.time_points)
            prices[symbol] = np.array(p)
        return prices

    def _process_data(self) -> np.ndarray:
        data_frames = []

        # Schritt 1: DataFrame für jedes Währungspaar erstellen und Indikatoren hinzufügen
        for symbol in self.trading_symbols:
            df = pd.DataFrame(self.prices[symbol], columns=['Open', 'Close', 'High', 'Low', 'Volume'])
            df_with_indicators = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close",
                                                     volume="Volume", fillna=True)
            data_frames.append(df_with_indicators)

        self.data_df = data_frames

        # Schritt 2: Bereinigen der DataFrames (Entfernen von NaNs oder andere Bereinigungen)
        #data_frames = [df.dropna() for df in data_frames]

        # Schritt 3: Zusammenführen der DataFrames
        combined_df = pd.concat(data_frames, axis=1)

        # Konvertieren des kombinierten DataFrame in ein numpy Array
        final_array = combined_df.to_numpy()

        return final_array



    def _get_observation(self) -> Dict[str, np.ndarray]:
        features = self.signal_features[(self._current_tick-self.window_size+1):(self._current_tick+1)]

        orders = np.zeros(len(self.trading_symbols))
        for i, symbol in enumerate(self.trading_symbols):
            order = self.simulator.symbol_orders(symbol)
            if len(order) > 0:
                order_type = order[0].type
                volume = order[0].volume
                if order_type == OrderType.Sell:
                    volume = -volume
                orders[i] = volume

        observation = {
            'balance': np.array([self.simulator.balance]),
            'equity': np.array([self.simulator.equity]),
            'margin': np.array([self.simulator.margin]),
            'features': features,
            'orders': orders,
        }
        return observation

    def _calculate_reward(self) -> float:
        prev_equity = self.history['equity'][-1]
        current_equity = self.simulator.equity
        step_reward = current_equity - prev_equity
        return step_reward

    def _calculate_reward(self) -> float:
        # Parameter für die Belohnungsberechnung
        gewicht_gewinn = 0.4
        gewicht_risiko = 0.3
        gewicht_verlustvermeidung = 0.1

        # Aktuelle und vorherige Equity-Werte
        aktuelle_equity = self.simulator.equity
        vorherige_equity = self.history['equity'][-1] if len(
            self.history['equity']) > 0 else 10000

        # Gewinnbasierte Belohnung
        gewinn = aktuelle_equity - vorherige_equity
        belohnung_gewinn = gewinn

        # Risikoadjustierte Belohnung (z.B. basierend auf dem Sharpe-Verhältnis)
        renditen = np.diff(self.history['equity'])
        if len(renditen) > 1:
            volatilitaet = np.std(renditen)
            if volatilitaet != 0:
                sharpe_verhaeltnis = np.mean(renditen) / volatilitaet
            else:
                sharpe_verhaeltnis = 0
        else:
            sharpe_verhaeltnis = 0
        belohnung_risiko = sharpe_verhaeltnis

        # Verlustvermeidung
        belohnung_verlustvermeidung = 0 if gewinn >= 0 else gewinn

        # Kombinierte Belohnung
        kombinierte_belohnung = gewicht_gewinn * belohnung_gewinn + \
                                gewicht_risiko * belohnung_risiko + \
                                gewicht_verlustvermeidung * belohnung_verlustvermeidung

        return kombinierte_belohnung

    def render(self, mode='human'):
        if mode not in self.metadata['render.modes']:
            raise Exception(f"Render mode {mode} not supported. Choose from {self.metadata['render.modes']}")

        if mode == 'human':
            # Einfache Konsolenausgabe
            print("Equity over Time:", self.history['equity'])
            print("Balance over Time:", self.history['balance'])
        elif mode in ['simple_figure', 'advanced_figure']:
            # Plotten von Equity und Balance in derselben Achse
            plt.figure(figsize=(10, 6))

            # Equity plotten
            plt.plot(self.history['equity'], label='Equity', color='blue')

            # Balance plotten
            plt.plot(self.history['balance'], label='Balance', color='green')

            # Titel, Beschriftungen und Legende
            plt.title('Equity and Balance over Time')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)

            if mode == 'advanced_figure':
                # Hier können erweiterte Plot-Funktionen hinzugefügt werden
                # Zum Beispiel zusätzliche Indikatoren oder Analysen
                pass

            # Anzeigen des Plots ohne das Skript zu blockieren und Aktualisieren der GUI
            plt.show(block=False)
            plt.pause(0.001)  # Wichtig für die Aktualisierung des Plots in Jupyter Notebooks

    def close(self) -> None:
        plt.close()
