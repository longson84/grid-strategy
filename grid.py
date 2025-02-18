#%%

import numpy as np
import copy
import logging

logging.basicConfig(level=logging.INFO)

class GridTrading():

    def __init__(self, prices, upper_range, lower_range, position_size=1e5, fee=0.01, num_grids=4, 
                 grid_type='bull', 
                 grid_levels_mode='linear', # linear = same absolute distance, percentage = same relative distance
                 sizing_mode='qty'):  # qty = same quantity, amount = same amount

        if num_grids < 3:
            raise ValueError("Number of grids must be at least 3.")
        
        if len(prices) < 3:
            raise ValueError('There must be at least 3 price points.')

        self.prices = prices
        self.position_size = position_size
        self.fee = fee*1e-2
        self.num_grids = num_grids
        self.grid_type = grid_type.lower()
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

        self.upper_range = upper_range
        self.lower_range = lower_range
    

        self.grid_levels_mode = grid_levels_mode
        self.grid_levels = self.calculate_grid_levels()

        logging.info(f'Grid levels: {self.grid_levels}')
        
        self.sizing_mode = sizing_mode

        self.amount = self.calculate_grid_amount() # will be used in case sizing_mode is 'amount'

        logging.info(f'Grid amount: {self.amount}')

        self.qty = self.calculate_grid_qty() # will be used in case sizing_mode is 'qty'

        logging.info(f'Grid qty: {self.qty}')
            
    def calculate_grid_qty(self):

        levels_above = len([g for g in self.grid_levels if g > self.current_price])

        s = sum([g for g in self.grid_levels if g < self.current_price]) + self.current_price * levels_above

        return self.position_size / s
    
    def calculate_grid_amount(self):

        return self.position_size / (self.num_grids - 1)

    def calculate_grid_levels(self):

        if self.grid_levels_mode == 'linear':
            return np.linspace(self.lower_range, self.upper_range, self.num_grids).tolist()

        elif self.grid_levels_mode == 'percentage':
            return np.exp(np.linspace(np.log(self.lower_range), np.log(self.upper_range), self.num_grids).tolist())
        
        else:  
            raise ValueError('Invalid grid levels mode. Choose either "linear" or "percentage".')
    
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
                    'Sell': [paired, self.qty, self.qty * paired, None]}
        
        elif self.sizing_mode == 'amount':
            qty = self.amount / price
            return {'Buy': [price, qty, self.amount, at], 
                    'Sell': [paired, qty, qty * paired, None]}

        else:
            raise ValueError('Invalid sizing mode. Choose either "qty" or "amount".')


    def init_first_trades(self):
        
        """Initialize the first set of trades based on the current price"""
        open_buy_orders = {}
        open_sell_orders = {}
        transactions = []
        fee = 0
        cash = self.position_size

        levels_below = [level for level in self.grid_levels if level < self.current_price]
        levels_above = [level for level in self.grid_levels if level > self.current_price]

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
        open_buy_orders = copy.deepcopy(self.open_buy_orders[prev]) # list of dictionary {Price: [qty, amount]}
        open_sell_orders = copy.deepcopy(self.open_sell_orders[prev]) # list of dictionary {Price: [qty, amount]}
        transactions = copy.deepcopy(self.transactions[prev])
        fee = 0
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

                    if self.sizing_mode == 'qty':
                        buy_qty = self.qty
                    elif self.sizing_mode == 'amount':
                        buy_qty = self.amount / new_buy_level
                    else:
                        raise ValueError('Invalid sizing mode')

                    open_buy_orders.update({
                        new_buy_level: [buy_qty, buy_qty * new_buy_level]
                    })


                # Update the transaction list with the filled sell order
                for trans in transactions:
                    if trans['Sell'][0] == level and trans['Sell'][3] is None:
                        trans['Sell'][3] = idx  # Update execution time
                        fee += self.fee * trans['Sell'][2]
                        cash += trans['Sell'][2] * (1 - self.fee)
                        grossed += trans['Sell'][2] - trans['Buy'][2]
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
MSFT = yfinance.download('MSFT', start='2022-01-01')[['Close']].squeeze().tolist()
TSLA = yfinance.download('TSLA', start='2022-01-01')[['Close']].squeeze().tolist()
AAPL = yfinance.download('AAPL', start='2022-01-01')[['Close']].squeeze().tolist()

BTC = yfinance.download('BTC-USD', start='2024-08-01')[['Close']].squeeze().tolist()
GOOGL = yfinance.download('GOOGL', start='2024-09-11')[['Close']].squeeze().tolist()





# %%
import matplotlib.pyplot as plt
from itertools import accumulate

pos = 1e5
grids = list(range(5, 50, 1))
lower_range = 270
upper_range = 600

grids = [20]

prices = MSFT

# prices = BTC
# lower_range = 50000
# upper_range = 120000
# grids = [71]

print(f'Buy and Hold: {pos * (prices[-1]/prices[0] - 1)}')

for grid in grids:

    grid_trader = GridTrading(prices=prices, 
                            lower_range=lower_range, 
                            upper_range=upper_range, 
                            position_size=pos,
                            fee=0,
                            num_grids=grid, 
                            grid_type='bull',
                            grid_levels_mode='percentage',
                            sizing_mode='qty')

    grid_trader.run()

    net_realized = list(accumulate(grid_trader.net_realized))
    pl = [realized + floating for realized, floating in zip(net_realized, grid_trader.floating)]

    buy_and_hold = [pos*(p/prices[0] - 1) for p in prices]
    
    x = range(len(net_realized))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1])

    # Plotting net_realized, floating, and total on the left y-axis
    ax1.plot(x, net_realized, label='Net Realized')
    ax1.plot(x, grid_trader.floating, label='Floating')
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

