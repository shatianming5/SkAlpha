'''
Module for manage portfolios.
'''
import pandas as pd
import numpy as np
np.seterr(divide='ignore')
# class PortfolioManager:
#     def __init__(self):
#         pass

#     def optimize_portfolio(self, stock_data, investment=1e7):
#         stock_data = pd.DataFrame(data={stock: data['Close'] for stock, data in stock_data.items()})

#         # Implement portfolio optimization logic based on the data
#         returns = stock_data.pct_change().dropna()  # Df: 2166 x 100
#         num_assets = len(returns.columns)  # 100
#         weights = np.ones(num_assets) / num_assets

#         # 最大化整个portfolio的：收益率/波动性
#         def objective_function(weights):
#             portfolio_returns = np.dot(np.asarray(returns), weights).mean()
#             portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
#             return -portfolio_returns / portfolio_volatility

#         # import pdb; pdb.set_trace()
#         constraints = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
#         bounds = tuple((0, 1) for _ in range(num_assets))  # bounds=bounds,

#         optimized_weights = minimize(objective_function, weights, bounds=bounds, method='SLSQP', constraints=constraints)
#         # optimized_weights = basinhopping(objective_function, weights, niter=10)

#         optimized_weights = optimized_weights.x
#         # 归一化，去除极小值

#         indices_sorted = optimized_weights.argsort()

#         # 方案1: 取20%股票，每支股票等额资金
#         indices_to_zero = indices_sorted[:-len(optimized_weights) // 3]
#         indices_to_one = indices_sorted[-len(optimized_weights) // 3:]
#         optimized_weights[indices_to_one] = np.linspace(3, 0.5, len(indices_to_one))
#         # import pdb; pdb.set_trace()
#         optimized_weights[indices_to_zero] = 0
#         optimized_weights /= stock_data.iloc[-1].values

#         # 方案2
#         # indices_to_one = indices_sorted[-len(optimized_weights)//10:]
#         # indices_to_zero = indices_sorted[:-len(optimized_weights)//10]
#         # optimized_weights[indices_to_one] = 1
#         # optimized_weights[indices_to_zero] = 0
#         # optimized_weights /= (optimized_weights.sum() * num_stocks_to_hold)

#         # 令投资总额=设定值
#         optimized_weights *= (investment * 0.8) / (optimized_weights).sum()
#         optimized_weights = np.floor(optimized_weights)

#         return optimized_weights


# class StockIndexPortfolioManager:
#     def __init__(self):
#         pass

#     def optimize_portfolio(self, dev_stock_data, test_stock_data, dev_index_data, test_index_data, investment=1e7, update_freq='M'):
#         if update_freq == 'M':
#             update_freq = 100
#         elif update_freq == 'S':
#             update_freq = 300
#         elif update_freq == 'Y':
#             update_freq = 10000
#         else:
#             raise NotImplementedError

#         dev_stock_data = pd.DataFrame(data={stock: data['Close'] for stock, data in dev_stock_data.items()})
#         test_stock_data = pd.DataFrame(data={stock: data['Close'] for stock, data in test_stock_data.items()})
#         all_stock_data = pd.concat([dev_stock_data, test_stock_data])

#         all_index_data = pd.concat([dev_index_data, test_index_data])
#         length_dev_data = len(dev_stock_data)

#         # 返回每日的portfolio（最高频），以适应不同更新频率
#         portfolio = pd.DataFrame(index=test_stock_data.index, columns=test_stock_data.columns)
#         last_year = test_stock_data.index[0] // update_freq - 1
#         for i in range(len(all_stock_data)):
#             if i >= length_dev_data and (all_stock_data.index[i] // update_freq) != last_year:
#                 stock = all_stock_data.iloc[i - length_dev_data:i]
#                 index = all_index_data.iloc[i - length_dev_data:i]
#                 time = all_stock_data.index[i]
#                 # 更新当前时刻往后的所有时间  投资额第一年开始每年线性增加50% max(1.0, i/480)    投资总额一段时间后指数增长（或按照实际收益来）
#                 portfolio.loc[time:] = self.optimize_portfolio_one_period(stock, index, investment=investment * max(1.0, -0.5 + (1.2) ** (1 + i / 240)))
#                 last_year = all_stock_data.index[i] // update_freq
#         return portfolio

#     def optimize_portfolio_one_period(self, stock_data: pd.DataFrame, index_data: pd.DataFrame, investment=1e7):
#         stock_data = pd.DataFrame(data={stock: data for stock, data in stock_data.items()})

#         # import pdb; pdb.set_trace()
#         # 算平均涨跌幅作为股指涨跌幅
#         index_pct_change = index_data.ffill(axis=0).pct_change().mean(axis=1)
#         relative_pct_change = stock_data.ffill(axis=0).pct_change() - np.expand_dims(index_pct_change, 1)

#         # 超额收益稳定性：
#         excess_return = relative_pct_change / relative_pct_change.std()

#         # excess_return = stock_data.ffill().pct_change() / relative_pct_change.std()
#         # excess_return = relative_pct_change.rolling(window=10, min_periods=1).max() \
#         #                               - relative_pct_change.rolling(window=10, min_periods=1).min()

#         optimized_weights = excess_return.mean(axis=0).values
#         optimized_weights = optimized_weights / optimized_weights.sum()

#         # 方案1: 取前1/3，每支股票等额资金 大到小
#         indices_sorted = optimized_weights.argsort()[::-1]
#         # 0-9
#         range_start = 1
#         range_end = 5

#         num_stocks = len(indices_sorted) // 10
#         indices_to_one = indices_sorted[range_start * num_stocks: range_end * num_stocks]
#         indices_to_zero = np.concatenate([
#             indices_sorted[:range_start * num_stocks],
#             indices_sorted[range_end * num_stocks:]
#         ])

#         # 分配等额资金到每支股
#         optimized_weights[indices_to_one] = 1.  # np.linspace(3, 0.5, len(indices_to_one))
#         optimized_weights[indices_to_zero] = 0.
#         optimized_weights /= stock_data.iloc[-1].values
#         # import pdb; pdb.set_trace()
#         # 去除nan
#         optimized_weights[np.isnan(optimized_weights)] = 0

#         # 令投资总额=设定值
#         # optimized_weights = np.ones(len(stock_data.columns)) # ###
#         optimized_weights *= (investment * 0.06) / (optimized_weights).sum()
#         # 变成100的倍数（手数）
#         optimized_weights = optimized_weights // 100 * 100
#         # 去除小于100股的数
#         optimized_weights[optimized_weights < 100] = 0

#         return optimized_weights

#     def optimize_portfolio_online(self, dev_stock_data: dict, test_stock_data: dict, index_data: pd.DataFrame, investment=1e7):
#         # import pdb; pdb.set_trace()
#         dev_stock_data_volume = pd.DataFrame(data={stock: data['Circulating MarketCap'] for stock, data in dev_stock_data.items()})
#         test_stock_data_volume = pd.DataFrame(data={stock: data['Circulating MarketCap'] for stock, data in test_stock_data.items()})
#         stock_data_vol = pd.concat([dev_stock_data_volume, test_stock_data_volume])

#         dev_stock_data_close = pd.DataFrame(data={stock: data['Close UnAdj'] for stock, data in dev_stock_data.items()})
#         test_stock_data_close = pd.DataFrame(data={stock: data['Close UnAdj'] for stock, data in test_stock_data.items()})
#         stock_data_close = pd.concat([dev_stock_data_close, test_stock_data_close])

#         # 取一年内的数据，根据交易额排序，取前1000支股票
#         stock_data_vol = stock_data_vol[stock_data_vol.index >= stock_data_vol.iloc[-1].name - 10000]

#         # 去除有NA的股票
#         stock_data_vol = stock_data_vol.loc[:, (~stock_data_vol.isna().any() & ~stock_data_close.isna().any())]
#         # 从大到小天取
#         stock_to_invest = np.argsort(stock_data_vol.mean())[-1000:].index

#         # 再从中选换手率高于1%的
#         dev_stock_data_turnover_rate = pd.DataFrame(data={stock: data['Turnover Rate'] for stock, data in dev_stock_data.items()})
#         test_stock_data_turnover_rate = pd.DataFrame(data={stock: data['Turnover Rate'] for stock, data in test_stock_data.items()})
#         stock_data_turnover_rate = pd.concat([dev_stock_data_turnover_rate, test_stock_data_turnover_rate])[stock_to_invest]
#         stock_to_invest = stock_to_invest[np.where(stock_data_turnover_rate.mean() > 2)[0]]
#         # import pdb; pdb.set_trace()

#         # stock_to_invest = np.argsort(stock_data_vol.mean())[:1000].index # 从小到大取

#         # 取筛选后股票的收盘价
#         stock_data_close = stock_data_close[stock_to_invest]
#         # 只取交易额前500的
#         # import pdb; pdb.set_trace()
#         # 算平均涨跌幅作为股指涨跌幅
#         index_pct_change = index_data.ffill(axis=0).pct_change().mean(axis=1)
#         relative_pct_change = stock_data_close.ffill(axis=0).pct_change() - np.expand_dims(index_pct_change, 1)

#         # 超额收益稳定性：
#         excess_return = relative_pct_change / relative_pct_change.std()
#         # excess_return = relative_pct_change.std()
#         # excess_return = stock_data.ffill().pct_change() / relative_pct_change.std()
#         # excess_return = relative_pct_change.rolling(window=10, min_periods=1).max() \
#         #                               - relative_pct_change.rolling(window=10, min_periods=1).min()

#         optimized_weights = excess_return.mean(axis=0).values
#         optimized_weights = optimized_weights / optimized_weights.sum()
#         # import pdb; pdb.set_trace()
#         # 方案1: 取前1/3，每支股票等额资金 大到小
#         indices_sorted = np.argsort(optimized_weights)[::-1]

#         # 依照排序分段取股票：0-9
#         range_start = 1
#         range_end = 6

#         num_frag = len(indices_sorted) // 10
#         indices_to_one = indices_sorted[range_start * num_frag: range_end * num_frag]
#         indices_to_zero = np.concatenate([
#             indices_sorted[:range_start * num_frag],
#             indices_sorted[range_end * num_frag:]
#         ])

#         # 分配等额资金到每支股
#         optimized_weights[indices_to_one] = np.linspace(2, 0.5, len(indices_to_one))  #
#         optimized_weights[indices_to_zero] = 0.
#         optimized_weights /= stock_data_close.iloc[-1].values
#         # import pdb; pdb.set_trace()
#         # 去除nan
#         # optimized_weights.fillna(0, inplace=True)

#         # 令投资总额=设定值
#         # optimized_weights = np.ones(len(stock_data.columns)) # ###
#         optimized_weights *= (investment * 0.8) / (optimized_weights).sum()
#         # 变成100的倍数（手数）
#         optimized_weights = optimized_weights // 100 * 100
#         # 去除小于100股的数
#         optimized_weights[optimized_weights < 100] = 0

#         # import pdb; pdb.set_trace()
#         all_stock_weights = pd.Series(np.zeros(len(dev_stock_data)), index=dev_stock_data_volume.columns)
#         all_stock_weights[stock_to_invest] = optimized_weights
#         return all_stock_weights

#     ### TO-DO: 迭代.step() 方法更新一天的交易，使用.query()来获取当前portfolio和持仓情况


class AlphaGPTPortfolioManager:
    def __init__(self, expr=None, update_freq='M', max_pos_each_stock=0.1):
        self.update_freq = update_freq
        assert self.update_freq in ['Y', 'HY', 'S', 'M'] or isinstance(self.update_freq, int)
        self.freq_mapping = {'Y': 12, 'HY': 6, 'S': 4, 'M': 1}
        self.recorded_timestamps = []
        self.day_count = 0
        
        self.max_pos_each_stock = max_pos_each_stock
        
    def whether_to_update_position(self, date):
        if self.update_freq in ['Y', 'HY', 'S', 'M']:
            yearmonth = f'{date.year}-{date.month//self.freq_mapping[self.update_freq]}'
            return yearmonth not in self.recorded_timestamps
        
        elif isinstance(self.update_freq, int):
            if self.day_count % self.update_freq == 0:
                self.day_count = 0
                return True
            else:
                return False
    
    def record_timestamp(self, date):
        if self.update_freq in ['Y', 'HY', 'S', 'M']:
            timestamp = f'{date.year}-{date.month//self.freq_mapping[self.update_freq]}'
            self.recorded_timestamps.append(timestamp)
        elif isinstance(self.update_freq, int):
            self.recorded_timestamps.append(date)
            self.day_count += 1
        else:
            raise NotImplementedError(f"不支持的调仓频率：{self.update_freq}")
    
    def optimize_portfolio_online(self, expr_var:dict=None, close_prices:pd.DataFrame=None, investment=1e7): 
        # 直接给定local数据
        self.alphaparser.safe_locals.update(expr_var)
        
        # self.alphaparser.safe_locals.update({
        #     'HIGH': data['HIGH']  ,
        #     'LOW': data['LOW']  ,
        #     'CLOSE': data['CLOSE']  ,
        #     'OPEN': data['OPEN']  ,
        #     'VOLUME': data['VOLUME']  ,
        #     'AMOUNT': data['AMOUNT']  ,
        #     'CAP': data['CAP']  ,
        #     'BENCHMARKINDEXOPEN': pd.DataFrame(data['BENCHMARKINDEXOPEN']).values  ,
        #     'BENCHMARKINDEXCLOSE': pd.DataFrame(data['BENCHMARKINDEXCLOSE']).values  ,
        #     'RET': data['RET']   , # data['CLOSE']  .pct_change(),
        #     'HD':  data['HIGH']   - DELAY(data['HIGH']  , 1),
        #     'LD': DELAY(data['LOW']  , 1) - data['LOW']  ,
        #  })
        
        # CLOSE = data['CLOSE']
        # OPEN = data['OPEN']
        # HIGH = data['HIGH']
        # LOW = data['LOW']
        # VOLUME = data['VOLUME']
        # AMOUNT = data['AMOUNT']
        # CAP = data['CAP']
        # BENCHMARKINDEXOPEN = data['BENCHMARKINDEXOPEN']
        # BENCHMARKINDEXCLOSE = data['BENCHMARKINDEXCLOSE']
        # RET = CLOSE.pct_change()
        # HD = HIGH - DELAY(HIGH, 1)
        # LD = DELAY(LOW, 1) - LOW
        
        # SUMIF(ABS(BENCHMARKINDEXCLOSE.mul(DELAY(BENCHMARKINDEXCLOSE,1), axis=0)).mul(AMOUNT, axis=0),20,CLOSE<DELAY(CLOSE,1))
        
        # import pdb; pdb.set_trace()
        
        # 执行表达式
        optimized_weights = eval(self.alpha_executable, 
                                 {"__builtins__": self.alphaparser.allowed_builtins}, 
                                 self.alphaparser.safe_locals)
        # import pdb; pdb.set_trace()
        # 处理不规范的输出
        if isinstance(optimized_weights, pd.DataFrame):
            optimized_weights = optimized_weights.iloc[-1]
            
        if optimized_weights.dtype == np.bool_:
            # import pdb; pdb.set_trace()
            indices_to_one = np.where(optimized_weights)[0]
            indices_to_zero = np.where(~optimized_weights)[0]
            optimized_weights = optimized_weights.astype(np.float32)
            normalized_alpha = (optimized_weights - optimized_weights.mean()) / (optimized_weights.std() + 1e-7)
            
            # print(optimized_weights.iloc[indices_to_one].index)
        else:
            # 正则化因子值
            optimized_weights = (optimized_weights - optimized_weights.mean()) / optimized_weights.std()
            normalized_alpha = optimized_weights.copy()
                
            optimized_weights.index = close_prices.columns
            optimized_weights = optimized_weights.fillna(-np.inf)
            
            # 依照因子值排序分层取股票：0-9 -> 大到小
            indices_sorted = np.argsort(optimized_weights.values)[::-1]
            range_start = 0
            range_end = 1

            num_frag = len(indices_sorted) // 10
            indices_to_one = indices_sorted[range_start * num_frag: range_end * num_frag]
            indices_to_zero = np.concatenate([
                indices_sorted[:range_start * num_frag],
                indices_sorted[range_end * num_frag:]
            ])

            # import pdb; pdb.set_trace()

            # 分配等额资金到每支股
            optimized_weights.iloc[indices_to_one] = 1.  # np.linspace(2, 0.5, len(indices_to_one)) #
            optimized_weights.iloc[indices_to_zero] = 0.
        # import pdb; pdb.set_trace()
        
        
        # 1除以股票价格，变成比例
        optimized_weights.iloc[indices_to_one] = optimized_weights.iloc[indices_to_one] / close_prices.iloc[-1, indices_to_one].values
        # import pdb; pdb.set_trace()
        # 去除nan
        # optimized_weights.fillna(0, inplace=True)
        num_stock_to_buy = len(indices_to_one)
        # 令投资总额=设定值
        optimized_weights *= min(investment/max(num_stock_to_buy, 1), investment*self.max_pos_each_stock) # 等额分配所有买入股票，并使得每支股票不超过总投资的10%
        
        # print((optimized_weights*close_prices.iloc[-1]).sum())
        # 变成100的倍数（多少手）
        optimized_weights = optimized_weights // 100 * 100
        # 去除小于100股的数
        optimized_weights[optimized_weights < 100] = 0

        return optimized_weights, normalized_alpha


def alpha_to_portfolio(alpha: pd.Series, current_prices: pd.Series, investment=1e7, layer=1, max_pos_each_stock=None):
    assert 1 <= layer < 10
    # 若因子不包含任何有效信息，则当期不买入
    # if (alpha.replace([np.nan, np.inf, -np.inf], 0) == 0).all():
    #     alpha = pd.Series(np.zeros_like(alpha.values, dtype=np.bool_), index=alpha.index)
    # 若是float类型的因子值，对因子截面进行标准化
    # alpha = alpha.sub(alpha.mean()).div(alpha.std())
    
    # 去除因子值或价格中的nan，确保portfolio中不会出现有nan的股票
    if alpha.dtype == np.bool_:
        alpha.loc[alpha.isna() | current_prices.isna()] = False
        indices_to_one = np.where(alpha)[0]
    else:
        alpha = alpha.astype('float32')
        alpha.loc[alpha.isna() | current_prices.isna()] = -np.inf
        # import pdb; pdb.set_trace()
        # 依照因子值排序分层取股票：0-9 -> 大到小
        indices_sorted = np.argsort(alpha.values)[::-1]
        # 选出来的index里面，在alpha里面是大于-inf的
        indices_sorted = indices_sorted[np.where(alpha.iloc[indices_sorted]!=-np.inf)[0]]
        range_start = layer - 1
        range_end = layer
        num_frag = len(indices_sorted) // 10
        indices_to_one = indices_sorted[range_start * num_frag: range_end * num_frag]
    
    # 选择持仓股
    portfolio = pd.Series(np.zeros([len(alpha)]), index=alpha.index) 
    portfolio.iloc[indices_to_one] = 1.
    # 根据当前股票价格算持仓比例, 1除以当前价格，得出需要购买股票的股数比例
    # import pdb; pdb.set_trace()
    # 平均分配的每支股的额度，不能占超过总投资额的一个比例, 且令投资总额<=设定值
    portfolio.iloc[indices_to_one] *= min(investment/max(len(indices_to_one), 1), investment*max_pos_each_stock)
    # 每支股票分配到的额度 / 当前股价 = 要购买的股数
    portfolio.iloc[indices_to_one] /= current_prices.iloc[indices_to_one]

    
    portfolio[portfolio.isna()] = 0
    
    # 变成100的倍数（多少手）
    portfolio = portfolio // 100 * 100
    # 去除小于100股的数
    portfolio[portfolio < 100] = 0
    
    normalized_alpha = alpha
    return portfolio, normalized_alpha