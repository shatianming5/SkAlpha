'''
Module for performance evaluation.
'''
from tqdm import tqdm
import numpy as np
import pandas as pd
from portfolio_manager.portfolio_management import AlphaGPTPortfolioManager, alpha_to_portfolio
from portfolio_manager.action_management import ActionManager

def calculate_win_to_loss_ratio(trade_signals, open_prices):
    win_count = 0
    total_trades = 0
    win_amount = 0
    loss_amount = 0

    for stock in trade_signals.columns:
        stock_signals = trade_signals[stock]
        stock_prices = open_prices[stock]
        buy_prices = []
        buy_volume = []
        for date, signal in stock_signals.items():
            if signal > 0:  # 买入信号
                # 记录购买成本和数量
                buy_prices.append(stock_prices[date])
                buy_volume.append(stock_signals[date])
            elif signal < 0 and len(buy_prices) > 0:  # 卖出信号
                # 卖出价格 / 平均购买价格
                sell_price = stock_prices[date]
                avg_buy_price = (np.array(buy_volume)*np.array(buy_prices)).sum() / np.array(buy_volume).sum()
                # 若卖出价格大于平均成本价
                if sell_price > avg_buy_price:
                    win_count += 1
                    win_amount += np.abs((sell_price - avg_buy_price) * signal)
                elif sell_price < avg_buy_price:
                    loss_amount += np.abs((sell_price - avg_buy_price) * signal)
                # 一次卖出算一次交易
                total_trades += 1
                
                # 若清仓，重置买入价/买入量。若未清仓，更新卖出后剩余股票的成本与数量
                if np.sum(buy_volume) + signal == 0:
                    buy_prices = []
                    buy_volume = []
                else:
                    buy_prices = [avg_buy_price]
                    buy_volume = [np.sum(buy_volume) + signal]

    if total_trades > 0:
        return np.round(win_count/total_trades, 2), np.round(win_amount/loss_amount, 2) 
        # f'{np.round(win_count/total_trades*100, 2)} % ({win_count} / {total_trades})', \
        # f'{np.round(win_amount/loss_amount, 2)} ({np.round(win_amount,0)} / {np.round(loss_amount,0)})'
    else:
        return 0, 0
    
def calculate_win_rate(trade_signals, open_prices):
    win_count = 0
    total_trades = 0

    for stock in trade_signals.columns:
        stock_signals = trade_signals[stock]
        stock_prices = open_prices[stock]
        buy_prices = []
        buy_volume = []
        for date, signal in stock_signals.items():
            if signal > 0:  # 买入信号
                # 记录购买成本和数量
                buy_prices.append(stock_prices[date])
                buy_volume.append(stock_signals[date])
            elif signal < 0 and len(buy_prices) > 0:  # 卖出信号
                # 卖出价格 / 平均购买价格
                sell_price = stock_prices[date]
                avg_buy_price = (np.array(buy_volume)*np.array(buy_prices)).sum() / np.array(buy_volume).sum()
                # 若卖出价格大于平均成本价
                if sell_price > avg_buy_price:
                    win_count += 1
                # 一次卖出算一次交易
                total_trades += 1
                
                # 若清仓，重置买入价/买入量。若未清仓，更新卖出后剩余股票的成本与数量
                if np.sum(buy_volume) + signal == 0:
                    buy_prices = []
                    buy_volume = []
                else:
                    buy_prices = [avg_buy_price]
                    buy_volume = [np.sum(buy_volume) + signal]
                
                
    
    if total_trades > 0:
        return np.round(win_count/total_trades, 3) #f'{np.round(win_count/total_trades*100, 2)} % ({win_count} / {total_trades})'
    else:
        return 0 #'0'

def calculate_info_ratio(excess_return: pd.Series, num_trading_days: int):
    return np.sqrt(num_trading_days) * excess_return.mean() / excess_return.std()


def calculate_max_drawdown(pnl: pd.Series):
    drawdowns = []
    prev_max_timestamps = []
    for i in range(len(pnl)):
        prev_max = max(pnl.iloc[:i+1])
        prev_max_timestamp = np.argmax(pnl.iloc[:i+1])
        prev_max_timestamps.append({'start': prev_max_timestamp, 'end': i})
        drawdown = (prev_max - pnl.iloc[i]) / prev_max
        drawdowns.append(drawdown)
    max_drawdown_idx = np.argmax(drawdowns)
    # 最大回撤起始结束时间
    start_idx = prev_max_timestamps[max_drawdown_idx]['start']
    end_idx = prev_max_timestamps[max_drawdown_idx]['end']
    start_timestamp = pnl.index[start_idx].split('T')[0]
    end_timestamp = pnl.index[end_idx].split('T')[0]
    return drawdowns[max_drawdown_idx], start_timestamp, end_timestamp


class PerformanceEvaluator:
    def __init__(self, risk_free_rate=0.0, num_trading_days=240):
        self.risk_free_rate = risk_free_rate
        self.num_trading_days = num_trading_days
        self.slippage = 0
        
    @staticmethod
    def _replace_nan(number):
        if isinstance(number, np.float64) and np.isnan(number):
            return 0.0
        elif number == np.inf or number == -np.inf:
            return 0.0
        else:
            return number

    def calculate_evaluation_metrics(self, results: dict, investment: float=1e7):
        # 读取、预处理回测结果
        expr = results['expr']
        portfolio = results['total_portfolios']
        trade_signals = results['trade_signals']
        
        xticks = np.asarray(trade_signals.index, dtype=str)
        pnl = results['pnl'].iloc[:,0]
        index = results['BENCHMARKINDEX']
        # index_open.index = pnl.index

        investment = results['start_cash']
        
        # 整理x轴标签，按年月分割
        xticks_yearmonth = []
        xticks_year = []
        added_yearmonth = []
        added_year = []
        
        for i, xtick in enumerate(xticks):
            yearmonth = xtick[:7]
            if yearmonth not in added_yearmonth:
                added_yearmonth.append(yearmonth)
                xticks_yearmonth.append(xtick)
            else:
                xticks_yearmonth.append('')
            
            year = xtick[:4]
            if year not in added_year or i == len(xticks) - 1: 
                added_year.append(year)
                xticks_year.append(xtick)
            else:
                xticks_year.append('')
        
        # 每日超额收益
        pnl = pnl
        roi = (pnl / investment).replace(np.inf, 0).replace(-np.inf, 0).fillna(0)
        index_roi = ((index / index.iloc[0]) - 1).replace([np.inf, -np.inf], 0).fillna(0).squeeze()
        index_roi.index = roi.index
        excess_return = (roi.values - index_roi.values)
        excess_return[np.isnan(excess_return)] = 0
        
        ym_excess_return = []
        # 月度超额收益率
        for i, ym in enumerate(xticks_yearmonth):
            if i == 0:
                ym_excess_return.append(0)
                last_ym = ym
                continue
            
            if ym != '':
                ym_excess_return.append(
                                    ((roi.loc[ym] - roi.loc[last_ym]) / (1+roi.loc[last_ym])) - \
                                    ((index_roi.loc[ym] - index_roi.loc[last_ym]) / (1+index_roi.loc[last_ym]))
                                    )
                last_ym = ym
            else:
                ym_excess_return.append(0)
            
        ym_excess_return = np.array(ym_excess_return)
        
        # 月度胜率
        sign = np.sign(ym_excess_return[np.where(ym_excess_return != 0)[0]])
        win = (sign == 1).sum()
        monthly_win_rate = np.round(win/len(sign), 3) ## f'{np.round(win/len(sign)*100, 2)} ({win} / {len(sign)})'
        
        
        # 月度超额收益bar宽度
        nonzero_idx = np.where(ym_excess_return!=0)[0]
        width = nonzero_idx - np.concatenate([np.zeros([1]), nonzero_idx[:-1]], 0)
        barwidth_fulllength = np.zeros_like(ym_excess_return)
        barwidth_fulllength[np.where(ym_excess_return!=0)[0]] = -width
        
        # 年度收益率 & 夏普率 & 最大回撤 & 交易胜率
        # years = []
        # y_roi = []
        # y_info_ratio = []
        # y_max_drawdown = []
        # y_trading_win_rate = []
        # for i, ym in enumerate(xticks_year):
        #     if i == 0:
        #         last_ym = ym
        #         continue
        #     if ym != '':
        #         # 策略/指数分别计算ROI
        #         roi_this_year = (pnl.loc[last_ym:ym]/(pnl.loc[last_ym]+investment)).fillna(0)
        #         index_roi_this_year = ((index_open.loc[last_ym:ym] / index_open.loc[last_ym]) - 1).fillna(0).squeeze()
        #         # 根据ROI计算指标
        #         years.append(f'{last_ym[:4]}-{ym[:4]}')
        #         y_roi.append(float(roi_this_year.iloc[-1]))
        #         y_info_ratio.append(float(calculate_info_ratio(excess_return=roi_this_year-index_roi_this_year)))
        #         y_max_drawdown.append(float(calculate_max_drawdown((investment+pnl.loc[last_ym: ym]).fillna(0))[0]))
        #         y_trading_win_rate.append(calculate_win_rate(trade_signals.loc[last_ym: ym], backtest_data_dict['OPEN'].loc[last_ym: ym]))
        #         last_ym = ym
        
        # 换手率: 平均（每日产生的交易额 / 每日总投资额）
        price = results['PRICE'] # 买入价
        turnover = ((trade_signals.abs() * price).replace(np.inf, 0).fillna(0).sum(axis=1) \
            / (portfolio * price).fillna(0).sum(axis=1)).replace([-np.inf, np.inf, np.nan], 0).mean(axis=0)
        turnover = np.round(turnover, 4) # f'{np.round(turnover*100, 2)} %'
        # 交易胜率
        overall_trading_win_rate, win_to_loss_ratio = calculate_win_to_loss_ratio(trade_signals, price)
        # 平均年化收益
        averaged_annualized_roi = np.round(float(roi.iloc[-1] / (len(xticks)/self.num_trading_days)), 4)
        # 信息比率
        info_ratio = np.round(calculate_info_ratio(excess_return=((1+roi).pct_change() - (1+index_roi).pct_change()).dropna(), 
                                                   num_trading_days=self.num_trading_days), 4)
        
        # 波动率
        ret = (investment+pnl).pct_change().fillna(0)
        index_ret = index.pct_change().fillna(0)
        algorithm_volatility = ret.std() * np.sqrt(self.num_trading_days/(len(ret)-1))
        algorithm_volatility = np.round(algorithm_volatility, 6)
        max_drawdown = np.round(calculate_max_drawdown(investment+pnl)[0], 4)
        
        # Alpha是投资者获得与市场波动无关的回报。比如投资者获得了15%的回报，其基准获得了10%的回报，那么Alpha或者价值增值的部分就是5%。
        index_annualized_roi = np.round(float(index_roi.iloc[-1] / (len(xticks)/self.num_trading_days)), 4)
        alpha = averaged_annualized_roi - index_annualized_roi
        alpha = np.round(alpha, 4)
        # Beta表示投资的系统性风险，反映了策略对大盘变化的敏感性
        beta = np.round(ret.cov(index_ret) / index_ret.var(), 4)
        
        xticks = [t.split('T00:00:00.')[0] for t in xticks]
        results = {
            'expr': expr,
            'date': xticks,
            # 每日指标
            'roi': roi.values.astype(float).tolist(),
            'index_roi': index_roi.values.astype(float).tolist(),
            'excess_return': excess_return.astype(float).tolist(),
            # 月度指标
            'xticks_yearmonth': [ym[:7] for ym in xticks_yearmonth],
            'barwidth_fulllength': barwidth_fulllength.tolist(),
            'ym_excess_return': ym_excess_return.astype(float).tolist(),
            # 年度指标
            # 'years': years,
            # 'y_roi': y_roi,
            # 'y_info_ratio': y_info_ratio,
            # 'y_max_drawdown': y_max_drawdown,
            # 'y_trading_win_rate': y_trading_win_rate,
            # 总体指标
            'averaged_annualized_roi': self._replace_nan(averaged_annualized_roi), # 平均年化收益率，字符串，如"12%"
            'info_ratio': self._replace_nan(info_ratio), # 信息比率，表示单位主动风险所带来的超额收益。
            'turnover': self._replace_nan(turnover), # 换手率，每日交易额占总持仓额的平均值
            'monthly_win_rate': self._replace_nan(monthly_win_rate), # 月度胜率，字符串，如"50% (18/36)"，等于盈利交易次数 / 总交易次数
            'overall_trading_win_rate': self._replace_nan(overall_trading_win_rate), # 总交易胜率，字符串，如"60% (3000/5000)
            'win_to_loss_ratio': self._replace_nan(win_to_loss_ratio), # 盈亏比，字符串，等于总盈利额 / 总亏损额
            'algorithm_volatility': self._replace_nan(algorithm_volatility), # 策略波动率
            'max_drawdown': self._replace_nan(max_drawdown),
            # 投资中面临着系统性风险（即Beta）和非系统性风险（即Alpha）
            'alpha': self._replace_nan(alpha), # Alpha，浮点数，是投资者获得与市场波动无关的回报。比如投资者获得了15%的回报，其基准获得了10%的回报，那么Alpha或者价值增值的部分就是5%。
            'beta': self._replace_nan(beta) # Beta，浮点数，表示投资的系统性风险，反映了策略对大盘变化的敏感性。Beta的计算方法为策略每日收益与基准每日收益的协方差除以基准每日收益的方差。
        }
        
        return results

    
    

    # def backtest(self, dev_stock_data, test_stock_data,
    #              dev_index, test_index,
    #              portfolio_manager, signal_generator,
    #              investment=1e7, update_freq='M'):
        
    #     # TO-DO: 单独计算每支股票的收益 / 月收益率 / 年收益率 ； 用简单的数据demo测试正确性
    #     test_open_unadj = pd.DataFrame(data={stock: data['Open UnAdj'] for stock, data in test_stock_data.items()})
    #     assert (test_open_unadj.index == test_index.index).all()
    #     test_ceiling = pd.DataFrame(data={stock: data['Ceiling'] for stock, data in test_stock_data.items()})
    #     test_floor = pd.DataFrame(data={stock: data['Floor'] for stock, data in test_stock_data.items()})
    #     allow_buying_signals = test_open_unadj < test_ceiling  # (pct_change <= 0.099).astype(int) # 涨停时暂停买入
    #     allow_selling_signals = test_open_unadj > test_floor  # (pct_change >= -0.099).astype(int) # 跌停时暂停卖出
    #     # import pdb; pdb.set_trace()

    #     test_open = pd.DataFrame(data={stock: data['Open PostAdj'] for stock, data in test_stock_data.items()})
    #     test_high = pd.DataFrame(data={stock: data['High PostAdj'] for stock, data in test_stock_data.items()})
    #     test_low = pd.DataFrame(data={stock: data['Low PostAdj'] for stock, data in test_stock_data.items()})
    #     test_close = pd.DataFrame(data={stock: data['Close PostAdj'] for stock, data in test_stock_data.items()})

    #     # 实际买入价每日差额，用于计算实际收益
    #     open_unadj_diff = test_open_unadj.diff()

    #     ##### 执行策略所需要的计算 #####
    #     close_pct_change = test_close.ffill().pct_change()
    #     vol = signal_generator.calculate_atr(test_high, test_low, test_close).rolling(window=signal_generator.dailyavg_m_ws, min_periods=1).mean()
    #     vol_ma = vol.rolling(window=signal_generator.dailyavg_l_ws, min_periods=1).mean()
    #     daily_avgs_s = test_close.rolling(window=signal_generator.dailyavg_s_ws, min_periods=1).mean()
    #     daily_avgs_m = test_close.rolling(window=signal_generator.dailyavg_m_ws, min_periods=1).mean()
    #     daily_avgs_m2 = test_close.rolling(window=signal_generator.dailyavg_m2_ws, min_periods=1).mean()
    #     daily_avgs_l = test_close.rolling(window=signal_generator.dailyavg_l_ws, min_periods=1).mean()
    #     daily_avgs_s_diff1 = daily_avgs_s.diff()
    #     # daily_avgs_s_diff2 = daily_avgs_s_diff1.diff().diff()
    #     # daily_avgs_w2_diff1 = daily_avgs_m.pct_change()
    #     daily_avgs_m_diff1 = daily_avgs_m.diff()
    #     # daily_avgs_m_diff2 = daily_avgs_m_diff1.diff()
    #     # daily_avgs_l_diff1 = daily_avgs_l.diff()
    #     daily_avgs_m_pctchange = daily_avgs_m.pct_change()

    #     std = test_close.rolling(window=signal_generator.vol_ws, min_periods=1).std().fillna(0)  # max 2.23   mean 0.14
    #     std_smooth = std.rolling(window=signal_generator.dailyavg_m_ws, min_periods=1).mean()  # max 2.22   mean 0.15
    #     # return std_smooth
    #     trend_s = daily_avgs_s_diff1.rolling(window=signal_generator.dailyavg_s_ws, min_periods=1).mean() / daily_avgs_s.rolling(window=signal_generator.dailyavg_s_ws, min_periods=1).mean()
    #     trend_m = daily_avgs_m_diff1.rolling(window=signal_generator.dailyavg_m_ws, min_periods=1).mean() / daily_avgs_m.rolling(window=signal_generator.dailyavg_m_ws, min_periods=1).mean()
    #     ##### END #####        

    #     ##### 初始化回测数据记录 #####
    #     portfolio = None
    #     pnl = []
    #     pnl_wo_svc = []
    #     daily_returns = []
    #     pnl_per_stock = []
    #     recorded = []
    #     actual_expenditure = []
    #     balance = investment
    #     # 记录每次买入数量，直到卖出信号清空
    #     buy_amount = np.zeros([len(test_open.columns)])
    #     position = np.zeros([len(test_open.columns)], dtype=int)
    #     holding_days = np.zeros([len(test_open.columns)])
    #     closing_days = np.zeros([len(test_open.columns)])
    #     close_pos_price = 9999999 * np.ones([len(test_open.columns)])
    #     open_pos_price = np.zeros([len(test_open.columns)])
    #     open_pos_std = np.zeros([len(test_open.columns)])

    #     total_portfolios = pd.DataFrame(np.zeros([len(test_open), len(test_open.columns)]), columns=test_open.columns, index=test_open.index)
    #     total_trade_signals = pd.DataFrame(np.zeros([len(test_open), len(test_open.columns)]), columns=test_open.columns, index=test_open.index)
    #     cumulative_trade_signals = pd.Series(np.zeros([len(test_open.columns)]), index=test_open.columns)
    #     bought_amount = pd.DataFrame(np.zeros([len(test_open), len(test_open.columns)]), columns=test_open.columns, index=test_open.index)

    #     # portfolio更新频率
    #     freq = {'Y': 10000, 'HY': 600, 'S': 400, 'M': 100}
    #     assert update_freq in freq.keys()
    #     reduction = 0.2
    #     max_iter = 1 // reduction
    #     # import pdb; pdb.set_trace()
    #     for i, date in enumerate(tqdm(test_open.index, desc="Backtesting")):
    #         if i == 0:
    #             pnl.append(0)
    #             zero_padding = pd.Series(np.zeros([len(test_open.columns)]), index=test_open.columns)
    #             pnl_per_stock.append(zero_padding)
    #             actual_expenditure.append(zero_padding)
    #             continue

    #         yearmonth = date // freq[update_freq]
    #         if yearmonth not in recorded:
    #             # 加载当前时刻股票总列表，并生成根据因子排序后的portfolio pd.concat([dev_close.iloc[:i], test_close.iloc[:i]])
    #             # import pdb; pdb.set_trace()
    #             portfolio = portfolio_manager.optimize_portfolio_online(
    #                 dev_stock_data={stock: data.iloc[i:] for stock, data in dev_stock_data.items()},  # 提取截至目前的数据
    #                 test_stock_data={stock: data.iloc[:i] for stock, data in test_stock_data.items()},
    #                 index_data=pd.concat([dev_index.iloc[i:], test_index.iloc[:i]]),
    #                 investment=investment * max(1.0, -0.1 + (1.2) ** (1 + i / 240))
    #             )

    #             portfolio = pd.Series(portfolio, index=test_open.columns)
    #             recorded.append(yearmonth)
    #             # import pdb; pdb.set_trace()

    #         # 生成交易信号
    #         # today_trade_signals = signal_generator.online_generate(test_high.iloc[i-1], test_low.iloc[i-1], test_close.iloc[i-1],
    #         #                                  position, holding_days, closing_days, 
    #         #                                  close_pos_price, open_pos_price, open_pos_std, 
    #         #                                  )
    #         # import pdb; pdb.set_trace()
    #         # test_open = test_open.iloc[i]

    #         if i < 10:  # 初始化，先不用信号
    #             today_trade_signals = pd.DataFrame(data=np.zeros([1, len(test_high.columns)], dtype=int),
    #                                                index=test_high.index[-2:-1],
    #                                                columns=test_high.columns, )
    #         else:
    #             factors = {
    #                 'test_close': test_close.iloc[:i],  # 昨收
    #                 'test_open': test_open.iloc[:i + 1],  # 今开
    #                 'close_pct_change': close_pct_change.iloc[:i],
    #                 'vol': vol.iloc[:i],
    #                 'vol_ma': vol_ma.iloc[:i],
    #                 'daily_avgs_s': daily_avgs_s.iloc[:i],
    #                 'daily_avgs_m': daily_avgs_m.iloc[:i],
    #                 'daily_avgs_m2': daily_avgs_m2.iloc[:i],
    #                 'daily_avgs_l': daily_avgs_l.iloc[:i],
    #                 'daily_avgs_s_diff1': daily_avgs_s_diff1.iloc[:i],
    #                 'daily_avgs_m_diff1': daily_avgs_m_diff1.iloc[:i],
    #                 'daily_avgs_m_pctchange': daily_avgs_m_pctchange.iloc[:i],
    #                 'std': std.iloc[:i],
    #                 'std_smooth': std_smooth.iloc[:i],
    #                 'trend_s': trend_s.iloc[:i],
    #                 'trend_m': trend_m.iloc[:i],
    #             }
    #             today_trade_signals = signal_generator.online_signal_generate(factors,
    #                                                                           position, holding_days, closing_days,
    #                                                                           close_pos_price, open_pos_price, open_pos_std,
    #                                                                           )

    #         today_trade_signals = today_trade_signals.iloc[-1]
    #         stock_bought = today_trade_signals > 0
    #         stock_sold = today_trade_signals < 0
    #         # TO-DO: 处理涨跌停
    #         # if (today_trade_signals < 0).any():
    #         #     import pdb; pdb.set_trace()
    #         # import pdb; pdb.set_trace()

    #         # 依据portfolio将tra de_signals转换实际买卖股数
    #         today_trade_signals.loc[stock_bought] = today_trade_signals.loc[stock_bought] * portfolio.loc[stock_bought] * allow_buying_signals.iloc[i].loc[stock_bought]
    #         today_trade_signals.loc[stock_sold] = -1 * buy_amount[stock_sold] * allow_selling_signals.iloc[i].loc[stock_sold]

    #         # 没钱就逐渐减少购买数量，直到不买
    #         j = 0
    #         stock_bought = today_trade_signals > 0
    #         stock_sold = today_trade_signals < 0
    #         today_buy_signals = today_trade_signals[stock_bought]
    #         while balance - (today_trade_signals * test_open_unadj.loc[date]).sum() <= 0 and j <= max_iter:
    #             today_trade_signals[stock_bought] -= reduction * today_buy_signals
    #             j += 1

    #         stock_bought = today_trade_signals > 0
    #         stock_sold = today_trade_signals < 0

    #         # 最后一日强制卖出
    #         if i == len(test_open) - 1:
    #             today_trade_signals.loc[stock_bought] = 0
    #             not_closed = position > 0
    #             today_trade_signals.loc[not_closed] = -1 * buy_amount[not_closed]

    #         # 记录每次买入实际数量
    #         buy_amount[stock_bought] += today_trade_signals.loc[stock_bought]
    #         buy_amount[stock_sold] = 0

    #         bought_amount.iloc[i] = bought_amount.iloc[i - 1]
    #         bought_amount.iloc[i].loc[stock_bought] += today_trade_signals.loc[stock_bought]
    #         bought_amount.iloc[i].loc[stock_sold] = 0

    #         if not (bought_amount.iloc[i] == buy_amount).all():
    #             import pdb;
    #             pdb.set_trace()
    #         # assert 

    #         # 记录开仓关仓价格，止损与开仓价有关，因此一段时间更新一次开仓价为一段时间内最高价来更新止损价
    #         stock_first_bought = stock_bought & (position == 0)
    #         open_pos_price[stock_first_bought] = test_close.iloc[i].loc[stock_first_bought]
    #         open_pos_std[stock_first_bought] = std.iloc[i].loc[stock_first_bought]
    #         close_pos_price[stock_sold] = test_close.iloc[i].loc[stock_sold]

    #         # 随时间更新开盘价（最近最高价），以更新止损线
    #         if i > 3 and i % 3 == 0:
    #             indices = np.where(open_pos_price != 0)[0]
    #             open_pos_price[indices] = test_close.iloc[i - 5: i + 1, indices].max()  # - 0.5*test_close.iloc[i-5: i+1, indices].std()

    #         # 记录仓位
    #         position[stock_bought] += 1
    #         position[stock_sold] = 0

    #         # 更新持仓关仓至今时间
    #         holding_days[stock_sold] = 0
    #         holding_days[position > 0] += 1
    #         closing_days[stock_bought] = 0
    #         closing_days[position == 0] += 1

    #         # 当日信号不能对当天收益产生影响, 因此用昨天更新后的持仓
    #         daily_pnl = cumulative_trade_signals * (open_unadj_diff.loc[date] + self.slippage)

    #         # if stock_bought.any():
    #         #     # import pdb; pdb.set_trace()
    #         #     print(f'Buy: {today_trade_signals.loc[stock_bought].index.values} \n',
    #         #           f'Num: {today_trade_signals.loc[stock_bought].values} \n' )
    #         # if stock_sold.any():
    #         #     print(f'Sell: {today_trade_signals.loc[stock_sold].index.values} \n',
    #         #           f'Num: {today_trade_signals.loc[stock_sold].values} \n' )
    #         print(f'{date} PnL: {np.sum(pnl)}')

    #         # 计算手续费
    #         service_charge = np.zeros([len(daily_pnl)])
    #         service_charge += (today_trade_signals > 0) * test_open_unadj.loc[date] * 0.00025  # 每日买入手续费：万分之2.5
    #         service_charge += (today_trade_signals < 0) * test_open_unadj.loc[date] * 0.00125  # 每日卖出手续费+印花税：万分之12.5

    #         # 累积信号：明日开盘开始计算收益
    #         cumulative_trade_signals += today_trade_signals.values
    #         # 计算现金余额的支出与收入
    #         daily_actual_expenditure = today_trade_signals * test_open_unadj.loc[date] - service_charge
    #         balance -= daily_actual_expenditure.sum()

    #         # 记录数据
    #         pnl_per_stock.append(daily_pnl - service_charge)
    #         pnl.append(daily_pnl.sum() - service_charge.sum())
    #         actual_expenditure.append(daily_actual_expenditure)
    #         total_portfolios.loc[date:] = np.expand_dims(portfolio, 0)
    #         total_trade_signals.loc[date] = today_trade_signals

    #     pnl = pd.DataFrame(pnl, index=np.asarray(test_open.index, dtype=str))
    #     pnl = pnl.cumsum()
    #     pnl_per_stock = pd.DataFrame(pnl_per_stock, index=np.asarray(test_open.index, dtype=str))
    #     pnl_per_stock = pnl_per_stock.cumsum()  # per-stock pnl
    #     # import pdb; pdb.set_trace()
    #     actual_expenditure = pd.DataFrame(actual_expenditure, index=np.asarray(test_open.index, dtype=str))
    #     # import pdb; pdb.set_trace()
    #     # 检查仓位是否归零

    #     results = {
    #         'pnl': pnl,
    #         'pnl_wo_svc': pnl_wo_svc,
    #         'daily_returns': daily_returns,
    #         # 'annualized_roi': annualized_roi, 
    #         # 'sharpe_ratio': sharpe_ratio,
    #         'pnl_per_stock': pnl_per_stock,
    #         'actual_expenditure': actual_expenditure,
    #         'bought_amount': bought_amount,
    #         'trade_signals': total_trade_signals,
    #         'total_portfolios': total_portfolios,
    #     }
    #     return results


    # def backtest_position_holding(self, backtest_data_dict, dev_stock_data, test_stock_data,
    #                               portfolio_manager: AlphaGPTPortfolioManager, 
    #                               action_manager: ActionManager):
        
    #     test_open_unadj = backtest_data_dict['OPEN']
    #     # import pdb; pdb.set_trace()
        
    #     # 前收盘价 | 开盘价 | 最高价 | 最低价 | 收盘价 | 成交量 | 成交金额 | 涨跌 | 涨跌幅 | 均价 | 换手率 | 总市值
    #     # TO-DO: 单独计算每支股票的收益 / 月收益率 / 年收益率 ； 用简单的数据demo测试正确性
    #     # import pdb; pdb.set_trace()
        
    #     test_close_lastday = backtest_data_dict['CLOSE'].shift(1)
    #     # TO-DO: 修改涨跌停判断条件
    #     test_ceiling = (test_close_lastday * 1.099).fillna(np.inf)
    #     test_floor = (test_close_lastday * 0.901).fillna(-np.inf)
    #     allow_buying_signals = test_open_unadj < test_ceiling  # (pct_change <= 0.099).astype(int) # 涨停时暂停买入
    #     allow_selling_signals = test_open_unadj > test_floor  # (pct_change >= -0.099).astype(int) # 跌停时暂停卖出
        
    #     # 以大写字母形式保存数据，以便因子计算
    #     TEST_DATA = test_stock_data
    #     DEV_DATA = dev_stock_data
        
    #     # import pdb; pdb.set_trace()
    #     # for k, v in self.stock_data_mapping.items():
    #     #     DEV_DATA.update({k: pd.DataFrame(data={stock: data[v] for stock, data in dev_stock_data.items()})})
    #     #     TEST_DATA.update({k: pd.DataFrame(data={stock: data[v] for stock, data in test_stock_data.items()})})

    #     test_open = backtest_data_dict['OPEN']
    #     # test_high = pd.DataFrame(data={stock: data['最高价'] for stock, data in test_stock_data.items()})
    #     # test_low = pd.DataFrame(data={stock: data['最低价'] for stock, data in test_stock_data.items()})
    #     test_close = backtest_data_dict['CLOSE']
        
    #     # 实际买入价每日差额，用于计算实际收益
    #     open_unadj_diff = test_open_unadj.diff()

    #     ##### 初始化回测数据记录 #####
    #     portfolio = None
    #     pnl = []
    #     # pnl_wo_svc = []
    #     pnl_per_stock = []
    #     recorded = []
    #     actual_expenditure = []
    #     balance = action_manager.start_cash
        
    #     action_manager.init_backtesting(example_data=test_open)
    #     # 记录每次买入数量，直到卖出信号清空
    #     # buy_amount = np.zeros([len(test_open.columns)])
        
    #     # position = np.zeros([len(test_open.columns)], dtype=int)
    #     # holding_days = np.zeros([len(test_open.columns)])
    #     # closing_days = np.zeros([len(test_open.columns)])
        
    #     # close_pos_price = 1e8 * np.ones([len(test_open.columns)])
    #     # open_pos_price = np.zeros([len(test_open.columns)])
    #     # open_pos_std = np.zeros([len(test_open.columns)])

    #     total_portfolios = pd.DataFrame(np.zeros([len(test_open), len(test_open.columns)]), columns=test_open.columns, index=test_open.index)
    #     total_trade_signals = pd.DataFrame(np.zeros([len(test_open), len(test_open.columns)]), columns=test_open.columns, index=test_open.index)
    #     cumulative_trade_signals = pd.Series(np.zeros([len(test_open.columns)]), index=test_open.columns)
    #     factor_values = pd.DataFrame(np.zeros([len(test_open), len(test_open.columns)]), columns=test_open.columns, index=test_open.index)

        
    #     pbar = tqdm(test_open.index)
    #     for i, date in enumerate(pbar):
    #         if i == 0:
    #             pnl.append(0)
    #             zero_padding = pd.Series(np.zeros([len(test_open.columns)]), index=test_open.columns)
    #             pnl_per_stock.append(zero_padding)
    #             actual_expenditure.append(zero_padding)
    #             continue
            
    #         if portfolio_manager.whether_to_update_position(date):
    #             # 加载当前时刻股票总列表，并生成根据因子排序后的portfolio
    #             today_trade_signals = pd.Series(data=np.zeros([len(test_close.columns)], dtype=int),
    #                                             name=test_close.index[i],
    #                                             index=test_close.columns, )
                
    #             # 记录当前portfolio，以便加仓减仓
    #             # if portfolio is not None:
    #             #     position = action_manager.position.copy()
    #             # else:
    #             #     position = pd.Series(np.zeros([len(test_close.columns)]), index=test_open.columns)
    #             # assert investment + np.sum(pnl) > -0.1 * investment
                
    #             INPUT_DATA = {}
    #             for k, v in DEV_DATA.items():
    #                 # 提取截至目前的数据
    #                 INPUT_DATA.update({
    #                     k: pd.concat([DEV_DATA[k].iloc[i:], TEST_DATA[k].iloc[:i]])
    #                 })
    #             # for k, v in self.index_data_mapping.items():
    #             #     INPUT_DATA.update({
    #             #         k: pd.concat([dev_index[v].iloc[i:], test_index[v].iloc[:i]])
    #             #     })
                
    #             portfolio, factor_value = portfolio_manager.optimize_portfolio_online(
    #                 expr_var=INPUT_DATA,
    #                 close_prices=test_close.iloc[i-1:i], # 前一天收盘价
    #                 investment=action_manager.calculate_current_investment(action_manager.start_cash+np.sum(pnl)),
    #                 )
                
    #             factor_values.loc[date:] = np.expand_dims(factor_value, 0)
    #             # import pdb; pdb.set_trace()
                
                
    #             portfolio.index = test_open.columns
    #             portfolio[np.isnan(test_close.iloc[i])] = 0 # 停牌的若之前有持仓，卖出
                
    #             # 加减仓
    #             to_hold = np.where(action_manager.position == portfolio)[0]
    #             to_buy = np.where(action_manager.position < portfolio)[0]
    #             to_sell = np.where(action_manager.position > portfolio)[0]

    #             today_trade_signals.iloc[to_hold] = 0
    #             today_trade_signals.iloc[to_buy] = 1
    #             today_trade_signals.iloc[to_sell] = -1
                
    #             # stock_bought = today_trade_signals > 0
    #             # stock_sold = today_trade_signals < 0

    #             # 依据portfolio将trade_signals转换实际买卖股数，考虑涨跌停
    #             today_trade_signals.iloc[to_buy] = (portfolio.iloc[to_buy] - action_manager.position.iloc[to_buy]) * allow_buying_signals.iloc[i].iloc[to_buy]
    #             today_trade_signals.iloc[to_sell] = (portfolio.iloc[to_sell] - action_manager.position.iloc[to_sell]) * allow_selling_signals.iloc[i].iloc[to_sell]
    #             today_trade_signals.fillna(0, inplace=True)
    #             # import pdb; pdb.set_trace()
    #         else:
    #             today_trade_signals = pd.Series(data=np.zeros([len(test_close.columns)], dtype=int),
    #                                             name=test_close.index[i],
    #                                             index=test_close.columns, )

    #             # 非调仓日，补足由于停牌、涨停等原因未买卖的部分
    #             today_trade_signals += portfolio - cumulative_trade_signals
    #             # 依据portfolio将trade_signals转换实际买卖股数，考虑涨跌停
    #             to_buy = np.where(today_trade_signals > 0)[0]
    #             to_sell = np.where(today_trade_signals < 0)[0]
    #             today_trade_signals.iloc[to_buy] *= allow_buying_signals.iloc[i].iloc[to_buy]
    #             today_trade_signals.iloc[to_sell] *= allow_selling_signals.iloc[i].iloc[to_sell]
    #             today_trade_signals.fillna(0, inplace=True)
            
    #         portfolio_manager.record_timestamp(date)
    #         # today_trade_signals = today_trade_signals.iloc[-1]
    #         # stock_bought = today_trade_signals > 0
    #         # stock_sold = today_trade_signals < 0
    #         # TO-DO: 处理涨跌停
    #         # if (today_trade_signals < 0).any():
    #         #     import pdb; pdb.set_trace()
            
            
            

    #         # 没钱就逐渐减少购买数量，直到不买
    #         # j = 0
    #         # stock_bought = today_trade_signals > 0
    #         # stock_sold = today_trade_signals < 0
    #         # today_buy_signals = today_trade_signals[stock_bought]
    #         # while balance - (today_trade_signals * test_open_unadj.loc[date]).sum() <= 0 and j <= max_iter:
    #         #     today_trade_signals[stock_bought] -= reduction * today_buy_signals
    #         #     j += 1

    #         # stock_bought = today_trade_signals > 0
    #         # stock_sold = today_trade_signals < 0

    #         # 最后一日强制卖出
    #         if i == len(test_open) - 1:
    #             today_trade_signals = pd.Series(data=np.zeros([len(test_close.columns)], dtype=int),
    #                         name=test_close.index[i],
    #                         index=test_close.columns)
    #             not_closed = action_manager.position > 0
    #             today_trade_signals.loc[not_closed] = -1 * action_manager.position[not_closed]

    #         # 处理止盈止损
    #         today_trade_signals = action_manager.process_trading_signals(today_trade_signals, test_open_unadj.loc[date])

    #         # 记录仓位、成本、开仓关仓时间
    #         action_manager.record_today_data(today_trade_signals, test_open_unadj.loc[date])
            

    #         # 当日信号不能对当天收益产生影响, 因此用昨天更新后的持仓
    #         daily_pnl = cumulative_trade_signals * (open_unadj_diff.loc[date] + self.slippage)
                
    #         # 计算手续费
    #         service_charge = np.zeros([len(daily_pnl)])
    #         service_charge += (today_trade_signals > 0) * test_open_unadj.loc[date] * 0.00025  # 每日买入手续费：万分之2.5
    #         service_charge += (today_trade_signals < 0) * test_open_unadj.loc[date] * 0.00125  # 每日卖出手续费+印花税：万分之12.5

    #         # 累积信号：明日开盘开始计算收益
    #         cumulative_trade_signals += today_trade_signals.values
    #         # 计算现金余额的支出与收入
    #         daily_actual_expenditure = today_trade_signals * test_open_unadj.loc[date] - service_charge
    #         balance -= daily_actual_expenditure.sum()
            
    #         # 记录数据
    #         pnl_per_stock.append(daily_pnl - service_charge)
    #         pnl.append(daily_pnl.sum() - service_charge.sum())
    #         actual_expenditure.append(daily_actual_expenditure)
    #         total_portfolios.loc[date:] = np.expand_dims(portfolio, 0)
    #         total_trade_signals.loc[date] = today_trade_signals
            
    #         # import pdb; pdb.set_trace()
    #         today_pnl = np.sum(pnl)
    #         today_pnl /= today_pnl + action_manager.start_cash
    #         pbar.set_description(f"Backtesting (PnL: {(100*today_pnl).round(2)} %)")

    #     pnl = pd.DataFrame(pnl, index=np.asarray(test_open.index, dtype=str))
    #     pnl = pnl.cumsum()
    #     pnl_per_stock = pd.DataFrame(pnl_per_stock, index=np.asarray(test_open.index, dtype=str))
    #     pnl_per_stock = pnl_per_stock.cumsum() # per-stock pnl
    #     actual_expenditure = pd.DataFrame(actual_expenditure, index=np.asarray(test_open.index, dtype=str))
    #     # import pdb; pdb.set_trace()
    #     # 检查仓位是否归零

    #     results = {
    #         'expr': portfolio_manager.expr, 
    #         'pnl': pnl, 
    #         'pnl_per_stock': pnl_per_stock,
    #         'actual_expenditure': actual_expenditure,
    #         'factor_values': factor_values,
    #         'trade_signals': total_trade_signals,
    #         'total_portfolios': total_portfolios,
    #     }

    #     return results

    

    def backtest_factor_table(self, alpha_table, deal_price_data, postadj_price_data, portfolio_manager: AlphaGPTPortfolioManager, action_manager: ActionManager):
        """
        portfolio_manager使用alpha_table生成仓位
        用deal_price_data计算开仓成本
        用postadj_price_data计算收益

        收盘前5分钟下单
        """
        ##### 初始化回测数据记录 #####
        assert (alpha_table.index == deal_price_data.index).all()
        portfolio = None
        pnl = [] # 每日pnl
        pnl_per_stock = [] # 每日pnl/stock
        pnl_wo_cost = [] # 每日pnl/stock
        actual_expenditure = [] # 每日实际支出
        balance = action_manager.start_cash # 现金余额
        
        action_manager.init_backtesting(example_data=deal_price_data)
        
        total_portfolios = pd.DataFrame(np.zeros([len(deal_price_data), len(deal_price_data.columns)]), columns=deal_price_data.columns, index=deal_price_data.index) # 每日持仓
        total_trade_signals = pd.DataFrame(np.zeros([len(deal_price_data), len(deal_price_data.columns)]), columns=deal_price_data.columns, index=deal_price_data.index) # 每日交易信号
        cumulative_trade_signals = pd.Series(np.zeros([len(deal_price_data.columns)]), index=deal_price_data.columns) # 累积交易信号，作为当前持仓
        normalized_alpha_table = pd.Series(np.zeros([len(deal_price_data.columns)]), index=deal_price_data.columns) # 正则化的因子表
        market_value = pd.Series(np.zeros([len(deal_price_data.columns)]), index=deal_price_data.columns) # 每日持仓市值
        # price_data_lastday = price_data.shift(-1)
        daily_return = postadj_price_data.pct_change(fill_method=None) # 每日收益（昨收——今收）
        allow_buying_signals = ~deal_price_data.isna() # 允许买入信号
        allow_selling_signals = ~deal_price_data.isna() # 允许卖出信号
        

        for i, date in enumerate(deal_price_data.index):
            # 初始化
            if i == 0:
                pnl.append(0)
                zero_padding = pd.Series(np.zeros([len(deal_price_data.columns)]), index=deal_price_data.columns)
                pnl_per_stock.append(zero_padding)
                actual_expenditure.append(zero_padding)
                continue
            
            # 判断是否调仓
            if portfolio_manager.whether_to_update_position(date):
                # 加载当前时刻股票总列表，并生成根据因子排序后的portfolio
                # 输入昨日收盘后的因子表和价格数据
                portfolio, normalized_alpha = alpha_to_portfolio(alpha=alpha_table.iloc[i], # 今日的因子表
                                                current_prices=deal_price_data.iloc[i], # 今日收盘价
                                                investment=action_manager.calculate_current_investment(action_manager.start_cash+np.sum(pnl)),
                                                layer=1,
                                                max_pos_each_stock=action_manager.max_pos_each_stock)
                
                portfolio.index = deal_price_data.columns
                # 停牌的若之前有持仓，卖出
                portfolio[np.isnan(deal_price_data.iloc[i])] = 0

                # 记录调仓信息（避免重复输出）
                if hasattr(self, '_last_rebalance_log') and self._last_rebalance_log != date:
                    self._last_rebalance_log = date
                elif not hasattr(self, '_last_rebalance_log'):
                    self._last_rebalance_log = date
                    print(f"开始回测，首次调仓日期：{date}，持仓股票数量：{(portfolio != 0).sum()}")
                
                # 加减仓
                to_hold = np.where(action_manager.position == portfolio)[0]
                to_buy = np.where(action_manager.position < portfolio)[0]
                to_sell = np.where(action_manager.position > portfolio)[0]

                # 构造本期交易信号
                today_trade_signals = pd.Series(data=np.zeros([len(deal_price_data.columns)], dtype=float),
                                name=deal_price_data.index[i],
                                index=deal_price_data.columns, )
                today_trade_signals.iloc[to_hold] = 0
                today_trade_signals.iloc[to_buy] = 1
                today_trade_signals.iloc[to_sell] = -1

                # 依据portfolio将trade_signals转换实际买卖股数，考虑涨跌停
                today_trade_signals.iloc[to_buy] = (portfolio.iloc[to_buy] - action_manager.position.iloc[to_buy]) * allow_buying_signals.iloc[i].iloc[to_buy]
                today_trade_signals.iloc[to_sell] = (portfolio.iloc[to_sell] - action_manager.position.iloc[to_sell]) * allow_selling_signals.iloc[i].iloc[to_sell]
                today_trade_signals.fillna(0, inplace=True)
                assert (today_trade_signals.iloc[to_sell] <= action_manager.position.iloc[to_sell]).all(), ValueError("卖出信号执行后，存在持仓数量将小于0的股票。")
                # import pdb; pdb.set_trace()
            else:
                today_trade_signals = pd.Series(data=np.zeros([len(deal_price_data.columns)], dtype=float),
                                                name=deal_price_data.index[i],
                                                index=deal_price_data.columns, )

                # 非调仓日，补足由于停牌、涨停等原因未买卖的部分
                today_trade_signals += portfolio - action_manager.position

                
            
            portfolio_manager.record_timestamp(date)
            # 最后一日强制卖出
            if i == len(deal_price_data) - 1:
                today_trade_signals = pd.Series(data=np.zeros([len(deal_price_data.columns)], dtype=float),
                            name=deal_price_data.index[i],
                            index=deal_price_data.columns)
                not_closed = action_manager.position > 0
                today_trade_signals.loc[not_closed] = -1 * action_manager.position[not_closed]
                
            # 处理止盈止损
            today_trade_signals, stock_stop_hold = action_manager.process_trading_signals(today_trade_signals, deal_price_data.loc[date])
            portfolio.loc[stock_stop_hold] = 0
            
            # 记录仓位、成本、开仓关仓时间
            action_manager.record_today_data(today_trade_signals, deal_price_data.loc[date])

            # 当天收益 = 昨日市值 * 昨收到今收的收益 ≈ 昨日持仓 * 昨日未复权价格 * 昨收到今收的收益 （考虑分红未考虑转股）
            daily_pnl = cumulative_trade_signals * deal_price_data.iloc[i-1] * daily_return.loc[date]

            # 累积信号，作为当前持仓
            cumulative_trade_signals += today_trade_signals

            buying_signals = today_trade_signals > 0 # 买入信号
            selling_signals = today_trade_signals < 0 # 卖出信号

            # holding_signals = today_trade_signals == 0 # 持仓信号
            # # 记录当前持仓市值
            # if i == 0:
            #     market_value = cumulative_trade_signals * deal_price_data.loc[date]
            # else:
            #     market_value[buying_signals] += deal_price_data.loc[date, buying_signals] * today_trade_signals[buying_signals]
            #     market_value[selling_signals] = 0
            #     market_value[holding_signals] += market_value[holding_signals] * daily_return.loc[date, holding_signals]
                
            # 计算手续费
            service_charge = pd.Series(data=np.zeros([len(deal_price_data.columns)], dtype=float),
                                                name=deal_price_data.index[i],
                                                index=deal_price_data.columns, )
            

            service_charge[buying_signals] += (today_trade_signals[buying_signals] * deal_price_data.loc[date, buying_signals] * (1 + self.slippage)) * 0.00025  # 每日买入手续费：万分之2.5
            service_charge[selling_signals] += (today_trade_signals[selling_signals].abs() * deal_price_data.loc[date, selling_signals] * (1 + self.slippage)) * 0.00125  # 每日卖出手续费+印花税：万分之12.5


            # 计算现金余额的支出与收入
            daily_actual_expenditure = today_trade_signals * deal_price_data.loc[date].fillna(0) - service_charge.fillna(0)
            balance -= daily_actual_expenditure.sum()
            
            # 记录数据
            pnl_per_stock.append(daily_pnl - service_charge)
            pnl.append(daily_pnl.sum() - service_charge.sum())
            pnl_wo_cost.append(daily_pnl.sum())
            actual_expenditure.append(daily_actual_expenditure)
            total_portfolios.loc[date:] = np.expand_dims(portfolio, 0)
            total_trade_signals.loc[date] = today_trade_signals
            normalized_alpha_table.loc[date] = normalized_alpha
            
            # 至今pnl
            today_pnl = np.sum(pnl)
            today_pnl /= action_manager.start_cash
            # pbar.set_description(f"Backtesting (PnL: {(100*today_pnl).round(2)} %)")

        pnl = pd.DataFrame(pnl, index=np.asarray(deal_price_data.index, dtype=str))
        pnl = pnl.cumsum()
        pnl_per_stock = pd.DataFrame(pnl_per_stock, index=np.asarray(deal_price_data.index, dtype=str))
        pnl_per_stock = pnl_per_stock.cumsum() # per-stock pnl
        actual_expenditure = pd.DataFrame(actual_expenditure, index=np.asarray(deal_price_data.index, dtype=str))
        results = {
            'pnl': pnl, 
            'pnl_per_stock': pnl_per_stock,
            'pnl_wo_cost': pnl_wo_cost,
            'actual_expenditure': actual_expenditure,
            'normalized_alpha_table': normalized_alpha_table,
            'trade_signals': total_trade_signals,
            'total_portfolios': total_portfolios,
        }

        return results