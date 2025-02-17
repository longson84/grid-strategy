#%%

import numpy as np
import logging
import copy

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GridTrading:
    
    def __init__(self, lower_range, higher_range, num_grids, position_size, prices, grid_type='bull', fee=0.1):
        self.lower_range = lower_range
        self.higher_range = higher_range
        self.num_grids = num_grids
        self.position_size = position_size
        self.prices = prices
        self.grid_type = grid_type.lower()
        self.fee = fee * 0.01  # Convert percentage to decimal
        self.grid_levels = np.linspace(lower_range, higher_range, num_grids).tolist()
        self.grid_distance = self.grid_levels[1] - self.grid_levels[0]
        self.current_price = prices[0]

        # Tracking variables
        self.open_buy_orders = []  # List of lists for open buy orders at each step
        self.open_sell_orders = []  # List of lists for open sell orders at each step
        self.transactions = []  # List of executed (buy, sell) pairs
        self.cash = []  # Cash at each step
        self.total_fee = []  # Cumulative fees
        self.gross_realized = []  # Gross realized profit
        self.net_realized = []  # Net realized profit after fees
        self.floating = []  # Floating P/L (unrealized profit/loss)

    def init_first_trades(self):
        """Initialize the first set of trades based on the current price"""
        open_buy_orders = []
        open_sell_orders = []
        transactions = []
        fee = 0
        cash = self.position_size

        levels_below = [level for level in self.grid_levels if level < self.current_price]
        levels_above = [level for level in self.grid_levels if level > self.current_price]

        if self.grid_type == "bull":
            self.N = self.position_size / (sum(levels_below) + self.current_price * (len(levels_above) - 1))
        elif self.grid_type == "bear":
            self.N = self.position_size / (sum(levels_above) + self.current_price * (len(levels_below) - 1))
        else:
            raise ValueError("grid_type must be either 'bull' or 'bear'")

        # Set up initial buy/sell orders
        for level in levels_below:
            open_buy_orders.append(level)

        for level in levels_above[1:]:
            open_sell_orders.append(level)
            transactions.append([[self.current_price, 0], [level, None]])  # Buy at current, sell above
            fee += self.fee * self.N * self.current_price
            cash -= self.N * self.current_price * (1 + self.fee)

        self.open_buy_orders.append(open_buy_orders)
        self.open_sell_orders.append(open_sell_orders)
        self.transactions.append(transactions)
        self.total_fee.append(fee)
        self.cash.append(cash)
        self.gross_realized.append(0)
        self.net_realized.append(-fee)
        self.floating.append(0)

    def process_price_point(self, idx):
        """Process a single price movement and update orders, cash, and P/L"""
        prev = idx - 1
        open_buy_orders = self.open_buy_orders[prev][:]
        open_sell_orders = self.open_sell_orders[prev][:]
        transactions = copy.deepcopy(self.transactions[prev])
        fee = 0
        cash = self.cash[prev]

        price = self.prices[idx]

        # Handle Buy Orders
        for level in self.open_buy_orders[prev][:]:  # Copy to avoid modifying list while iterating
            if price <= level:
                open_buy_orders.remove(level)  # Remove filled buy order
                new_sell_level = level + self.grid_distance
                if new_sell_level <= self.higher_range:
                    open_sell_orders.append(new_sell_level)  # Place paired sell order
                transactions.append([[level, idx], [new_sell_level, None]])  # Track execution
                fee += self.fee * self.N * level
                cash -= self.N * level * (1 + self.fee)

        # Handle Sell Orders
        grossed = 0
        net = 0
        for level in self.open_sell_orders[prev][:]:
            if price >= level:
                open_sell_orders.remove(level)  # Remove filled sell order
                grossed += self.N * self.grid_distance # gross profit
                
                new_buy_level = level - self.grid_distance
                if new_buy_level >= self.lower_range:
                    open_buy_orders.append(new_buy_level)  # Place paired buy order
                
                for trans in transactions:
                    if trans[1][0] == level and trans[1][1] is None:
                        trans[1] = [level, idx]  # Update execution time
                        break
                
                fee += self.fee * self.N * level
                cash += self.N * level * (1 - self.fee)
        
        net = grossed - fee


        # Floating P/L Calculation
        floating = sum(
            (price - trans[0][0]) * self.N
            for trans in transactions
            if trans[1][1] is None  # Only count open positions
        )

        self.open_buy_orders.append(open_buy_orders)
        self.open_sell_orders.append(open_sell_orders)
        self.transactions.append(transactions)
        self.total_fee.append(self.total_fee[prev] + fee)
        self.cash.append(cash)
        self.gross_realized.append(grossed)
        self.net_realized.append(net)
        self.floating.append(floating)

    def run(self):
        """Run the grid trading simulation across all price points"""
        self.init_first_trades()

        for i in range(1, len(self.prices)):
            self.process_price_point(i)

        



# %%
import yfinance

# Example usage
MSFT = yfinance.download('MSFT', start='2024-08-04')[['Close']].squeeze().tolist()
BTC = yfinance.download('BTC-USD', start='2024-11-1')[['Close']].squeeze().tolist()
GOOGL = yfinance.download('GOOGL', start='2024-09-11')[['Close']].squeeze().tolist()




# %%
import matplotlib.pyplot as plt
from itertools import accumulate

pos = 1e5
grids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50]
lower_range = 140
higher_range = 200
instrument = GOOGL

print(f'Buy and Hold: {pos * (instrument[-1]/instrument[0] - 1)}')

for grid in grids:

    grid_trader = GridTrading(prices=instrument, 
                            lower_range=lower_range, 
                            higher_range=higher_range, 
                            num_grids=grid, 
                            grid_type='bull',
                            position_size=pos,
                            fee=0)

    grid_trader.run()

    net_realized = list(accumulate(grid_trader.net_realized))
    pl = [realized + floating for realized, floating in zip(net_realized, grid_trader.floating)]

    # x = range(len(net_realized))

    # fig, ax1 = plt.subplots(figsize=(16, 4))

    # # Plotting net_realized, floating, and total on the left y-axis
    # ax1.plot(x, net_realized, label='Net Realized')
    # ax1.plot(x, grid_trader.floating, label='Floating')
    # ax1.plot(x, pl, label='Total')

    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Net Realized, Floating, Total', color='tab:blue')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')
    # ax1.grid(True)

    # # Creating the right y-axis to plot the price (MSFT)
    # ax2 = ax1.twinx()
    # ax2.plot(x, MSFT, label='Price', color='tab:red')
    # ax2.set_ylabel('Price', color='tab:red')
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    # # Displaying the legend
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    # plt.show()


    print(f"{grid} grids, Net realized: {net_realized[-1]:.2f}, Floating: {grid_trader.floating[-1]:.2f}, Total: {pl[-1]:.2f}")

    


# %%
