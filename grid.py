#%%

import numpy as np
import copy
import logging

logging.basicConfig(level=logging.INFO)

class Grid():

    def __init__(self, upper_range, lower_range, grid_levels_mode='linear'):
        
        self.upper_range = upper_range
        self.lower_range = lower_range
        self.grid_levels_mode = grid_levels_mode
        self.grid_levels = []
        
    
class Grid_Space(Grid):

    def __init__(self, upper_range, lower_range, grid_space=10, grid_levels_mode='linear'):

        super().__init__(upper_range, lower_range, grid_levels_mode)
        self.grid_space = grid_space
        self.grid_levels = self._get_grid_levels()

    def _get_grid_levels(self):
        if self.grid_levels_mode == 'linear':
            return np.arange(start=self.lower_range, 
                             stop=self.upper_range + self.grid_space, 
                             step=self.grid_space).tolist()
        elif self.grid_levels_mode == 'percentage':
            space = 1 + 1e-2*self.grid_space
            log_levels = np.arange(start=np.log(self.lower_range), 
                                   stop=np.abs(np.log((space) * self.upper_range)), 
                                   step=np.log(space))
            return np.exp(log_levels).tolist()
        else:
            raise ValueError('grid_levels_mode must be either linear or percentage')
        
class Grid_Number(Grid):
    
    def __init__(self, upper_range, lower_range, num_grids=4, grid_levels_mode='linear'):

        super().__init__(upper_range, lower_range, grid_levels_mode)

        self.num_grids = num_grids
        self.grid_levels = self._get_grid_levels()
    
    def _get_grid_levels(self):
        if self.grid_levels_mode == 'linear':
            return np.linspace(self.lower_range, self.upper_range, self.num_grids).tolist()
        elif self.grid_levels_mode == 'percentage':
            return np.geomspace(self.lower_range, self.upper_range, self.num_grids).tolist()
        else:
            raise ValueError("Invalid grid_levels_mode. Must be 'linear' or 'percentage'.")
    


class Grid_Trading():

    def __init__(self, prices, 
                 grid: Grid,
                 position_size=1e5, 
                 fee=0.1, # fee in percentage
                 sizing_mode='qty'):  # qty = same quantity, amount = same amount

        if len(prices) < 3:
            raise ValueError('There must be at least 3 price points.')

        self.prices = prices
        self.position_size = position_size
        self.fee = fee*1e-2
        self.grid = grid
        self.initial_price = prices[0]

        self.upper_range = grid.upper_range
        self.lower_range = grid.lower_range    

        self.grid_levels_mode = grid.grid_levels_mode
        self.grid_levels = grid.grid_levels
        
        self.sizing_mode = sizing_mode
        self.amount = self.calculate_grid_amount() # will be used in case sizing_mode is 'amount'
        self.qty = self.calculate_grid_qty() # will be used in case sizing_mode is 'qty'
        
        # Tracking variables
        self.open_buy_orders = []  # List of lists for open buy orders at each step
        self.open_sell_orders = []  # List of lists for open sell orders at each step
        self.transactions = []  # List of executed (buy, sell) pairs
        self.cash = []  # Cash at each step
        self.total_fee = []  # Cumulative fees
        self.gross_realized = []  # Gross realized profit
        self.net_realized = []  # Net realized profit after fees
        self.floating = []  # Floating P/L (unrealized profit/loss)
        self.invested = [] # amount invested at each step

    def update_state(self, 
                     open_buy_orders, 
                     open_sell_orders,
                     transactions,
                     fee, cash, grossed, net, floating, invested):

        self.open_buy_orders.append(open_buy_orders)
        self.open_sell_orders.append(open_sell_orders)
        self.transactions.append(transactions)
        self.total_fee.append(fee)
        self.cash.append(cash)
        self.gross_realized.append(grossed)
        self.net_realized.append(net)
        self.floating.append(floating)
        self.invested.append(invested)
        

    def calculate_grid_qty(self):
        levels_above = len([g for g in self.grid_levels if g > self.initial_price])
        s = sum([g for g in self.grid_levels if g < self.initial_price]) + self.initial_price * levels_above
        return self.position_size / s
    
    def calculate_grid_amount(self):
        return self.position_size / (self.num_grids - 1)
    
    def closest_level_above(self, price):
        """Find the closest grid level above the given price"""
        if price >= self.upper_range:
            return 0
        
        return min([level for level in self.grid_levels if level > price])
                   
    def closest_level_below(self, price):
        """Find the closest grid level below the given price"""
        if price <= self.lower_range:
            return 0

        return max([level for level in self.grid_levels if level < price])
    
    def open_order_at_level(self, price):

        if self.sizing_mode == 'qty':
            return {price: [self.qty, self.qty * price]}
        elif self.sizing_mode == 'amount':
            return {price: [self.amount / price, self.amount]}
        else:
            raise ValueError('Invalid sizing mode. Choose either "qty" or "amount".')
        
    def execute_trade_at_level(self, price, paired=0, at=0):

        if paired == 0:
            paired = self.closest_level_above(price)

        if self.sizing_mode == 'qty':
            return {'Buy': [price, self.qty, self.qty * price, at], 
                    'Sell': [paired, self.qty, self.qty * paired, None],
                    'Grossed': 0
                    }        
        elif self.sizing_mode == 'amount':
            qty = self.amount / price
            return {'Buy': [price, qty, self.amount, at], 
                    'Sell': [paired, qty, qty * paired, None],
                    'Grossed': 0}
        else:
            raise ValueError('Invalid sizing mode. Choose either "qty" or "amount".')


    def init_first_trades(self):
        
        """Initialize the first set of trades based on the current price"""
        open_buy_orders = {}
        open_sell_orders = {}
        transactions = []
        fee = 0
        cash = self.position_size

        levels_below = [level for level in self.grid_levels if level < self.initial_price]
        levels_above = [level for level in self.grid_levels if level > self.initial_price]

        current_price = self.prices[0]

        # Set up initial buy/sell orders
        for level in levels_below:
            open_buy_orders.update(self.open_order_at_level(level))

        for level in levels_above[1:]:
            open_sell_orders.update(self.open_order_at_level(level))
            
            trade = self.execute_trade_at_level(price=current_price, paired=level)
            transactions.append(trade)  # Buy at current, sell above
            
            fee += self.fee * trade['Buy'][2]
            cash -= trade['Buy'][2]

        invested = self.position_size - cash

        self.update_state(open_buy_orders=open_buy_orders,
                          open_sell_orders=open_sell_orders,
                          transactions=transactions,
                          fee=fee, 
                          cash=cash, 
                          grossed=0, 
                          neted=-fee, 
                          floating=-fee, 
                          invested=invested)

    

    def process_price_point(self, idx):
        """Process a single price movement and update orders, cash, and P/L"""
        prev = idx - 1
        open_buy_orders = copy.deepcopy(self.open_buy_orders[prev]) # list of dictionary {Price: [qty, amount]}
        open_sell_orders = copy.deepcopy(self.open_sell_orders[prev]) # list of dictionary {Price: [qty, amount]}
        transactions = copy.deepcopy(self.transactions[prev])
        fee = self.total_fee[prev]
        cash = self.cash[prev]

        price = self.prices[idx]

        # Handle Buy Orders
        for level in self.open_buy_orders[prev].keys():  # Copy to avoid modifying list while iterating
            if price <= level:
                open_buy_orders.pop(level)  # Remove filled buy order
                
                # Form the pair of filled buy order and a new sell order
                trade = self.execute_trade_at_level(price=level, at=idx)

                # Add this new open sell order to the list of open sell order
                sell_order = trade['Sell']
                open_sell_orders.update({
                    sell_order[0]: [sell_order[1], sell_order[2]]
                    })

                # Add the paired order to the list of transactions
                transactions.append(trade)  # Track execution

                # Calculating fee and cash
                fee += self.fee * trade['Buy'][2]
                cash -= trade['Buy'][2]

        # Handle Sell Orders
        grossed = 0
        net = 0

        for level in self.open_sell_orders[prev].keys():
            if price >= level:
                open_sell_orders.pop(level)  # Remove filled sell order
                
                # Placing the new buy order at the closest level below the filled sell order
                new_buy_level = self.closest_level_below(level)
                if new_buy_level > 0:
                    open_buy_orders.update(self.open_order_at_level(new_buy_level))

                # Update the transaction list with the filled sell order
                for trans in transactions:
                    if trans['Sell'][0] == level and trans['Sell'][3] is None:
                        trans['Sell'][3] = idx  # Update execution time
                        fee += self.fee * trans['Sell'][2]
                        cash += trans['Sell'][2] * (1 - self.fee)
                        grossed += trans['Sell'][2] - trans['Buy'][2]
                        trans['Grossed'] = grossed
                        break
            
        net = grossed - fee


        # Floating P/L Calculation
        holding_amount = sum(
            trans['Buy'][2]
            for trans in transactions
            if trans['Sell'][3] is None  # Only count open positions 
        )

        holding_qty = sum(
            trans['Buy'][1]
            for trans in transactions
            if trans['Sell'][3] is None  # Only count open positions
        )

        floating = price * holding_qty - holding_amount

        self.update_state(
            open_buy_orders=open_buy_orders,
            open_sell_ỏders=open_sell_orders,
            transactions=transactions,
            fee=fee,
            cash=cash,
            grossed=grossed,
            net=net,
            floating=floating,
            invested=holding_amount
        )

    def run(self):
        """Run the grid trading simulation across all price points"""
        self.init_first_trades()

        for i in range(1, len(self.prices)):
            self.process_price_point(i)

        

# %%
import yfinance

start_date = '2022-01-01'


# Example usage
MSFT = yfinance.download('MSFT', start=start_date)[['Close']].squeeze().tolist()
# TSLA = yfinance.download('TSLA', start=start_date)[['Close']].squeeze().tolist()
# AAPL = yfinance.download('AAPL', start=start_date)[['Close']].squeeze().tolist()
# META = yfinance.download('META', start=start_date)[['Close']].squeeze().tolist()
# GOOGL = yfinance.download('GOOGL', start=start_date)[['Close']].squeeze().tolist()
# AMZN = yfinance.download('AMZN', start=start_date)[['Close']].squeeze().tolist()



# BTC = yfinance.download('BTC-USD', start=start_date)[['Close']].squeeze().tolist()
# ETH = yfinance.download('ETH-USD', start=start_date)[['Close']].squeeze().tolist()

# %%
import matplotlib.pyplot as plt
from itertools import accumulate

pos = 1e5

grid_number_list = [20]

prices = MSFT

lower_range = prices[0]*0.8
upper_range = prices[0]*2

# prices = BTC
# lower_range = 50000
# upper_range = 120000
# grids = [71]

print(f'Buy and Hold: {pos * (prices[-1]/prices[0] - 1)}')


for g in grid_number_list:

    grid = Grid_Number(
        lower_range=lower_range,
        upper_range=upper_range,
        num_grids=g,
        grid_levels_mode='linear')

    grid_trader = Grid_Trading(prices=prices,
                            position_size=pos,
                            grid=grid,
                            fee=0,
                            sizing_mode='amount')

    grid_trader.run()

    net_realized = list(accumulate(grid_trader.net_realized))
    pl = [realized + floating for realized, floating in zip(net_realized, grid_trader.floating)]

    buy_and_hold = [pos*(p/prices[0] - 1) for p in prices]
    
    x = range(len(net_realized))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1])

    # Plotting net_realized, floating, and total on the left y-axis
    ax1.plot(x, net_realized, label='Net Realized')
    # ax1.plot(x, grid_trader.floating, label='Floating')
    ax1.plot(x, pl, label='Total')
    ax1.plot(x, buy_and_hold, label='Buy and Hold')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Net Realized, Floating, Total', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)


    # Creating the right y-axis to plot the price (MSFT)
    # ax2 = ax1.twinx()
    # ax2.plot(x, prices, label='Price', color='tab:red')
    # ax2.set_ylabel('Price', color='tab:red')
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    # Displaying the legend
    ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    # Plotting the cash level on the second subplot (ax2)
    ax2.plot(x, grid_trader.cash, label='Cash Level', color='tab:green')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cash Level', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.grid(True)
    ax2.legend(loc='upper left')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()

    trans = grid_trader.transactions
    buys = grid_trader.open_buy_orders
    sells = grid_trader.open_sell_orders
    levels = grid_trader.grid_levels
    fees = grid_trader.total_fee


    print(f"{grid} grids, Net realized: {net_realized[-1]:.2f}, Floating: {grid_trader.floating[-1]:.2f}, Total: {pl[-1]:.2f}")

    


# %%
 
