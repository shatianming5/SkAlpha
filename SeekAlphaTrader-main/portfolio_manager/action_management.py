import numpy as np
import pandas as pd

class ActionManager:
    """
    在已有portfolio_manager基础上进一步加入仓位控制、止盈止损等操作。
    """
    def __init__(self, **actions):
        """
        Params:
        - start_cash: 用于股票投资的初始总资金
        - position_size: 对整个投资组合中所投入的资金量在总资金中的占比
        """

        self.stop_loss_rate = actions.get('stop_loss_rate', 0.3)
        self.stop_profit_rate = actions.get('stop_profit_rate', 0.3)
        self.start_cash = actions.get('start_cash', 1e7)
        self.position_size = actions.get('position_size', 0.9)
        
        self.update_freq = actions.get('update_freq', 'M')
        self.max_pos_each_stock=actions.get('max_pos_each_stock', 0.1)
        self.stock_pool=actions.get('stock_pool', "沪深两市")
                
        for item in [self.stop_loss_rate, self.stop_profit_rate, self.position_size, self.max_pos_each_stock]:
            if item is not None:
                item = float(item)
                assert item >= 0 and item <= 1
        if self.start_cash is not None:
            assert self.start_cash >= 1e6
        if self.update_freq is not None:
            assert self.update_freq in ['Y', 'HY', 'S', 'M'] or isinstance(self.update_freq, int)

        
    def __repr__(self) -> str:
        return "Actions: [Start Cash: {:.1f}], [position_size: {:.2f}%], [Stop Loss: {:.2f}%], [Stop Profit: {:.2f}%]".format(self.start_cash, self.position_size*100, self.stop_loss_rate*100, self.stop_profit_rate*100)
        
    def init_backtesting(self, example_data):
        self.avg_cost = pd.Series(np.zeros([len(example_data.columns)]), index=example_data.columns)
        self.position = pd.Series(np.zeros([len(example_data.columns)], dtype=int), index=example_data.columns)
        self.holding_days = pd.Series(np.zeros([len(example_data.columns)], dtype=int), index=example_data.columns)
        self.closing_days = pd.Series(np.zeros([len(example_data.columns)], dtype=int), index=example_data.columns)
            
    def record_today_data(self, today_trade_signals, open_prices):
        """
        记录每支股票不计交易手续费的购买成本
        """
        traded_stock = today_trade_signals != 0 # 需要买入/加仓/减仓/卖出的股票
        stock_to_buy = today_trade_signals > 0 # 买入/加仓的部分
        stock_to_reduce = (today_trade_signals < 0) & ((today_trade_signals - today_trade_signals) > 0) # 减仓但未平仓的部分
        stock_to_hold = stock_to_buy | stock_to_reduce
        stock_to_close = (today_trade_signals < 0) & ((today_trade_signals - today_trade_signals) == 0)
        
        if traded_stock.sum() != 0:
            # 记录买入/加仓股票的成本
            self.avg_cost[stock_to_hold] = (self.position[stock_to_hold]*self.avg_cost[stock_to_hold] + open_prices[traded_stock]*today_trade_signals[traded_stock]) \
                / (self.position[stock_to_hold] + today_trade_signals[stock_to_hold])
            self.avg_cost[stock_to_close] = 0
            
            # 记录仓位变化
            self.position[traded_stock] = self.position[traded_stock] + today_trade_signals[traded_stock]
            # 仓位不能为负
            assert (self.position >= 0).all()
        
        self.holding_days[today_trade_signals < 0] = 0
        self.holding_days[self.position > 0] += 1
        self.closing_days[today_trade_signals > 0] = 0
        self.closing_days[self.position == 0] += 1
    
    def calculate_current_investment(self, value):
        return self.position_size * value
    
    def process_trading_signals(self, today_trade_signals, open_prices):
        # 止损卖出
        if self.stop_loss_rate is not None:
            stock_stop_loss = (open_prices < (1 - self.stop_loss_rate) * self.avg_cost) & (self.position > 0)
            if stock_stop_loss.sum() > 0:
                today_trade_signals[stock_stop_loss] = -self.position[stock_stop_loss]
            
        # 止盈卖出
        if self.stop_profit_rate is not None:
            stock_stop_profit = (open_prices > (1 + self.stop_profit_rate) * self.avg_cost) & (self.position > 0)
            if stock_stop_profit.sum() > 0:
                today_trade_signals[stock_stop_profit] = -self.position[stock_stop_profit]
                
        stock_stop_hold = stock_stop_profit | stock_stop_loss
        return today_trade_signals, stock_stop_hold

    