'''
Module for visualizing stock data and backtesting results.
'''
import logging
logging.basicConfig(level=logging.WARNING)

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import datetime
import os
import json
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 优先使用这些字体
plt.rcParams['axes.unicode_minus'] = True  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

def visualize_single_stock_prices_and_signals(xticks_shown, stock_data, trade_signals, pnl_per_stock, portfolio, sma_win_size, indices_to_vis, foldername):
    
    xticks = pnl_per_stock.index.astype(str)
    # import pdb; pdb.set_trace()
    # indices_to_vis = np.random.permutation(indices_to_vis)
    indices_to_vis = np.argsort(np.abs(pnl_per_stock.sum().values))[::-1] # 收益从大到小
    stocks_to_vis = np.array([code for (code, _) in stock_data.items()])[indices_to_vis]
    
    # open = pd.DataFrame(data={stock: data['Open PostAdj'] for stock, data in stock_data.items()})
    # # import pdb; pdb.set_trace()
    # stocks_to_vis = np.array(open.loc[:, (np.abs(pnl_per_stock.sum().values>100))&(open.isna().sum() > 100)].columns)
    
    
    for i, code in enumerate(stocks_to_vis):
        plt.figure(figsize=(5*len(xticks)//1500 + 15, 5*len(xticks)//1500 + 15,))
        
        # 图1
        plt.subplot(2, 1, 1)
        plt.title(f'Single Stock Profit and Loss (PnL) Over Time. ') # 
        ax1 = plt.gca()
        # 在 ax1 上绘制 PnL 数据
        plt.plot(pnl_per_stock.index.astype(str), pnl_per_stock[code].values, label=f'{code} PnL with service charges', color='orange')
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.xticks(xticks_shown, rotation=70, size=6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.twinx()
        ax2 = plt.gca()
        plt.plot(xticks, portfolio.loc[:, code], label='Position given by the portfolio manager')
        plt.ylim(0, 4/3*portfolio.loc[:, code].max())
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        # plt.legend()

        # 图2: 均线，开盘价
        plt.subplot(2, 1, 2)
        for w in sma_win_size: 
            dailyavg = stock_data[code]['Close PostAdj'].rolling(window=w, min_periods=1).mean()
            if w != sma_win_size[-1]:
                plt.plot(xticks, dailyavg.values, label=f'Daily average close price of {code}, windows size = {w}', \
                    linewidth=2, zorder=1)
            
            elif w == sma_win_size[-1]: 
                std = stock_data[code]['Close PostAdj'].rolling(window=60, min_periods=1).std().rolling(window=12, min_periods=1).mean()
                # avg_m = stock_data[code]['Close'].rolling(window=6, min_periods=1).mean()
                # trend = avg_m.diff().rolling(window=6, min_periods=1).mean() / avg_m.rolling(window=6, min_periods=1).mean()
                # std = std*(4/5-trend*10)
                plt.plot(xticks, dailyavg.values, label=f'Daily average close price of {code}, windows size = {w}', \
                    linewidth=2, zorder=1)
                
                # daily_ema = stock_data[code]['Close'].ewm(span=w, adjust=False).mean()
                # plt.plot(xticks, daily_ema.values, label=f'Daily EMA price of {code}, windows size = {w}', \
                # linewidth=1, zorder=1)
                plt.fill_between(xticks, (dailyavg - std), (dailyavg + std), color='gray', alpha=0.2)
                
        plt.plot(xticks, stock_data[code]['Open PostAdj'].values, label=f'Daily Open price of {code}', \
            linewidth=1, zorder=2, color='green')

        plt.ylabel('Stock Price')
        
        
        # 图二：可视化买卖信号
        x_index = trade_signals[code].reset_index()
        # import pdb; pdb.set_trace()
        x_buy_date = x_index[x_index[code] > 0]['Date']
        y_buy_price = stock_data[code]['Open PostAdj'].loc[x_buy_date]
        x_buy = np.asarray(x_buy_date.index)
        y_buy = y_buy_price.values
        plt.scatter(x_buy, y_buy, color='black', s=35, marker='o', zorder=2, label='Acutal Buy')
        
        x_sell_date = x_index[x_index[code] < 0]['Date']
        y_sell_price = stock_data[code]['Open PostAdj'].loc[x_sell_date]
        x_sell = np.asarray(x_sell_date.index)
        y_sell = y_sell_price.values
        plt.scatter(x_sell, y_sell, color='blue', s=35, marker='o', zorder=2, label='Acutal Sell')
        plt.xticks(xticks_shown, rotation=90, size=5)
        ax1 = plt.gca()

        plt.twinx()
        ax2 = plt.gca()
        
        # 方案1：可视化交易量
        # plt.plot(xticks, stock_data[code]['Volume'].values, label=f'Daily Volume of {code}', \
        #     linewidth=1, zorder=2, color='blue')
        # plt.ylim(0, 3*stock_data[code]['Volume'].max())
        
        # 方案2： 可视化ATR波动率
        # high = pd.DataFrame({code: stock_data[code].loc[:,'High']})
        # low = pd.DataFrame({code: stock_data[code].loc[:,'Low']})
        # close = pd.DataFrame({code: stock_data[code].loc[:,'Close']})
        # atr = LowHighestPointStrategy().calculate_atr(high, low, close).rolling(window=6, min_periods=1).mean()
        # plt.plot(xticks, atr.values, label=f'Average True Range (volatility rate)', \
        #     linewidth=1, zorder=2, color='orange')
        # plt.ylim(0, 0.1)
        
        # 方案3：可视化真实持仓情况
        plt.plot(xticks, trade_signals[code].cumsum().values, label='Position')
        plt.ylim(0, 4/3*trade_signals[code].cumsum().max())
        plt.ylabel('Position')
        # 方案4：可视化趋势
        # trend_m = (dailyavg.diff() / dailyavg).rolling(window=w, min_periods=1).mean()
        # plt.plot(xticks, trend_m.values, 'r', label='trend')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.savefig(f'./figures/{foldername}/{code}.png')
        
        # 加仓股票
        # if len(x_index[x_index[code] < -1]) != 0:
        #     print(code)


def visualize_porfolio():
    pass







def visualize(stock_data, trade_signals, optimized_portfolio, indices, results, foldername='', window_size=[], investment=1e7):
    # 预处理
    trade_signals = trade_signals.shift(1)
    actual_trade_signals = results['trade_signals']
    pnl = results['pnl']
    # pnl_wo_svc = results['pnl_wo_svc']
    # daily_returns = results['daily_returns']
    # annualized_roi = results['annualized_roi']
    # sharpe_ratio = results['sharpe_ratio']
    pnl_per_stock = results['pnl_per_stock']
    xticks = np.asarray(trade_signals.index, dtype=str)
    
    
    # 设置x轴标签
    xticks_yearmonth = []
    xticks_year = []
    added_yearmonth = []
    added_year = []
    anuual_return = []
    anuual_return_rate = []
    for i, xtick in enumerate(xticks):
        yearmonth = xtick[:6]
        if yearmonth not in added_yearmonth:
            added_yearmonth.append(yearmonth)
            xticks_yearmonth.append(xtick)
        else:
            xticks_yearmonth.append('')
        
        year = xtick[:4]
        if year not in added_year: 
            added_year.append(year)
            xticks_year.append(xtick)
            anuual_return.append(float(pnl.loc[xtick].values))
        else:
            xticks_year.append('')
            
        if i == len(xticks) - 1:
            anuual_return.append(float(pnl.loc[xtick].values))
    
    for i, profit in enumerate(anuual_return):
        if i == 0:
            anuual_return_rate.append(float(profit/1e7))
        else:
            anuual_return_rate.append(float((profit-anuual_return[i-1])/(anuual_return[i-1] + investment) ))
    anuual_return_rate = anuual_return_rate[1:]
    
    string_roi = ''
    for i in range(len(anuual_return_rate)):
        if i < len(anuual_return_rate) - 1:
            string_roi += '\n{}-{}: {}'.format(added_year[i], added_year[i+1], np.round(anuual_return_rate[i]*100, 2))
        else:
            string_roi += '\n{}: {}'.format(added_year[i], np.round(anuual_return_rate[i]*100, 2))  
    
    
    # import pdb; pdb.set_trace()
    # 绘制Portfolio综合收益曲线
    plt.figure(figsize=(5*len(xticks)//1500 + 10, 5*len(xticks)//1500 + 6,))
    plt.title(f'Porfolio Profit and Loss (PnL) Over Time. \
        \n Investment: {investment} \
        \n Annualized RoI: {string_roi} %' )
        # Sharpe Ratio: {round(100*sharpe_ratio, 4)} %')

    plt.plot(xticks, pnl.values, label='Portfolio PnL', color='blue')
    # plt.plot(xticks, pnl_wo_svc.values, label='Portfolio PnL without service charges', color='orange')
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.xticks(xticks_yearmonth, rotation=80, size=6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    
    # 保存图表
    if not os.path.exists(f'./figures/{foldername}'):
        os.makedirs(f'./figures/{foldername}')
    plt.savefig(f'./figures/{foldername}/Portfolio_PNL.png')
    
    
    # 可视化现金持有
    plt.figure(figsize=(5*len(xticks)//1500 + 10, 5*len(xticks)//1500 + 10,))
    plt.title(f'Currency held overtime. \
        \n Investment: {investment}')
    plt.plot(xticks, investment-results['actual_expenditure'].sum(axis=1).cumsum())
    plt.xlabel('Date')
    plt.ylabel('Currency Held')
    plt.xticks(xticks_yearmonth, rotation=80, size=6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(f'./figures/{foldername}/Portfolio_Held_Currency.png')
    
    
    
    # 可视化持仓情况
    plt.figure(figsize=(5*len(xticks)//1500 + 10, 5*len(xticks)//1500 + 10,))
    plt.title(f'Stock position overtime. \
        \n Investment: {investment}')
    plt.plot(xticks, investment-results['actual_expenditure'].sum(axis=1).cumsum())
    plt.xlabel('Date')
    plt.ylabel('Currency Held')
    plt.xticks(xticks_yearmonth, rotation=80, size=6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(f'./figures/{foldername}/Portfolio_Held_Currency.png')
    
    
    
    indices = np.random.permutation(indices)
    stocks_to_vis = np.array([code for (code, _) in stock_data.items()])[indices]
    for i, code in enumerate(stocks_to_vis):
        plt.figure(figsize=(5*len(xticks)//1500 + 15, 5*len(xticks)//1500 + 15,))
        
        # 图1
        plt.subplot(2, 1, 1)
        plt.title(f'Single Stock Profit and Loss (PnL) Over Time. ') # 
        
        # 在 ax1 上绘制 PnL 数据
        plt.plot(xticks, pnl_per_stock[code].values, label=f'{code} PnL with service charges', color='orange')
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.xticks(xticks_yearmonth, rotation=70, size=6)
        plt.axhline(y=0, color='r', linestyle='--')
        
        # import pdb; pdb.set_trace()
        
        # 重整portfolio数据
        # timestamps = []
        # portfolios = []
        # for time, portfolio in optimized_portfolio.items():
        #     timestamps.append(time)
        #     portfolios.append(portfolio)
        # timestamps = np.array(timestamps, dtype=str)
        # portfolios = np.stack(portfolios)[:,indices]
        # plt.bar(timestamps, portfolio[:,indices][:,i], width=100, align='edge')
        
        # import pdb; pdb.set_trace()
        plt.plot(optimized_portfolio.index.astype(str), optimized_portfolio.iloc[:, indices[i]])
        # 画线
        # for j, time in enumerate(timestamps):
        #     if j == 0:
        #         plt.hlines(portfolios[j,i], xticks[0], timestamps[j+1])
        #         plt.vlines(timestamps[j+1], portfolios[j,i], portfolios[j+1,i])
        #     elif j > 0 and j < len(optimized_portfolio) - 1:
        #         plt.hlines(portfolios[j,i], timestamps[j], timestamps[j+1])
        #         plt.vlines(timestamps[j+1], portfolios[j,i], portfolios[j+1,i])
        #     else:
        #         plt.hlines(portfolios[j,i], timestamps[j], xticks[-1], label='portfolio weight')
            
        plt.legend()
        # plt.tick_params(axis='y', labelcolor='blue')


        # 图2
        plt.subplot(2, 1, 2)
        for w in window_size: 
            dailyavg = stock_data[code]['Close PostAdj'].rolling(window=w, min_periods=1).mean()
            if w != window_size[-1]:
                plt.plot(xticks, dailyavg.values, label=f'Daily average close price of {code}, windows size = {w}', \
                    linewidth=2, zorder=1)
            
            elif w == window_size[-1]: 
                std = stock_data[code]['Close PostAdj'].rolling(window=60, min_periods=1).std().rolling(window=12, min_periods=1).mean()
                # avg_m = stock_data[code]['Close'].rolling(window=6, min_periods=1).mean()
                # trend = avg_m.diff().rolling(window=6, min_periods=1).mean() / avg_m.rolling(window=6, min_periods=1).mean()
                # std = std*(4/5-trend*10)
                plt.plot(xticks, dailyavg.values, label=f'Daily average close price of {code}, windows size = {w}', \
                    linewidth=2, zorder=1)
                
                # daily_ema = stock_data[code]['Close'].ewm(span=w, adjust=False).mean()
                # plt.plot(xticks, daily_ema.values, label=f'Daily EMA price of {code}, windows size = {w}', \
                # linewidth=1, zorder=1)
                plt.fill_between(xticks, (dailyavg - std), (dailyavg + std), color='gray', alpha=0.2)
                
        plt.plot(xticks, stock_data[code]['Open PostAdj'].values, label=f'Daily Open price of {code}', \
            linewidth=1, zorder=2, color='green')

        plt.ylabel('Stock Price')
        # # 可视化买卖信号
        # x_index = trade_signals[code].reset_index()
        # x_buy_date = x_index[x_index[code] > 0]['Date']
        # y_buy_price = stock_data[code]['Open'].loc[x_buy_date]
        # x_buy = np.asarray(x_buy_date.index)
        # y_buy = y_buy_price.values
        # plt.scatter(x_buy, y_buy, color='black', s=25, marker='^', zorder=1, label='Buy', alpha=0.5)
        # # import pdb; pdb.set_trace()
        
        # x_sell_date = x_index[x_index[code] < 0]['Date']
        # y_sell_price = stock_data[code]['Open'].loc[x_sell_date]
        # x_sell = np.asarray(x_sell_date.index)
        # y_sell = y_sell_price.values
        # plt.scatter(x_sell, y_sell, color='blue', s=25, marker='v', zorder=1, label='Sell', alpha=0.5)
        # plt.xticks(xticks_yearmonth, rotation=75, size=5)
        # plt.legend(loc='upper left')
        
        
        # 可视化买卖信号
        x_index = actual_trade_signals[code].reset_index()
        # import pdb; pdb.set_trace()
        x_buy_date = x_index[x_index[code] > 0]['Date']
        y_buy_price = stock_data[code]['Open'].loc[x_buy_date]
        x_buy = np.asarray(x_buy_date.index)
        y_buy = y_buy_price.values
        plt.scatter(x_buy, y_buy, color='black', s=35, marker='o', zorder=2, label='Acutal Buy')
        # import pdb; pdb.set_trace()
        
        x_sell_date = x_index[x_index[code] < 0]['Date']
        y_sell_price = stock_data[code]['Open PostAdj'].loc[x_sell_date]
        x_sell = np.asarray(x_sell_date.index)
        y_sell = y_sell_price.values
        plt.scatter(x_sell, y_sell, color='blue', s=35, marker='o', zorder=2, label='Acutal Sell')
        plt.xticks(xticks_yearmonth, rotation=90, size=5)
        ax1 = plt.gca()

        plt.twinx()
        ax2 = plt.gca()
        # 方案1：可视化交易量
        # plt.plot(xticks, stock_data[code]['Volume'].values, label=f'Daily Volume of {code}', \
        #     linewidth=1, zorder=2, color='blue')
        # plt.ylim(0, 3*stock_data[code]['Volume'].max())
        
        # 方案2： 可视化ATR波动率
        # high = pd.DataFrame({code: stock_data[code].loc[:,'High']})
        # low = pd.DataFrame({code: stock_data[code].loc[:,'Low']})
        # close = pd.DataFrame({code: stock_data[code].loc[:,'Close']})
        # atr = LowHighestPointStrategy().calculate_atr(high, low, close).rolling(window=6, min_periods=1).mean()
        # plt.plot(xticks, atr.values, label=f'Average True Range (volatility rate)', \
        #     linewidth=1, zorder=2, color='orange')
        # plt.ylim(0, 0.1)
        
        # 方案3：可视化真实持仓情况
        plt.plot(xticks, actual_trade_signals[code].cumsum().values, label='Position')
        plt.ylim(0, 4/3*actual_trade_signals[code].cumsum().max())
        
        # 方案4：可视化趋势
        # trend_m = (dailyavg.diff() / dailyavg).rolling(window=w, min_periods=1).mean()
        # plt.plot(xticks, trend_m.values, 'r', label='trend')

        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.savefig(f'./figures/{foldername}/{code}.png')
        
        if len(x_index[x_index[code] < -1]) != 0:
            print(code)






def visualize_strategy(stock_data, results, foldername='', sma_win_size=[], investment=1e7):
    # 读取，预处理
    portfolio = results['total_portfolios']
    trade_signals = results['trade_signals']
    pnl = results['pnl']
    pnl_per_stock = results['pnl_per_stock']
    bought_amount = results['bought_amount']
    indices_to_vis = np.where((trade_signals*portfolio != 0).any(axis=0).values)[0]
    xticks = np.asarray(trade_signals.index, dtype=str)
    
    # 设置x轴标签
    xticks_yearmonth = []
    xticks_year = []
    added_yearmonth = []
    added_year = []
    anuual_return = []
    anuual_return_rate = []
    for i, xtick in enumerate(xticks):
        yearmonth = xtick[:6]
        if yearmonth not in added_yearmonth:
            added_yearmonth.append(yearmonth)
            xticks_yearmonth.append(xtick)
        else:
            xticks_yearmonth.append('')
        
        year = xtick[:4]
        if year not in added_year: 
            added_year.append(year)
            xticks_year.append(xtick)
            anuual_return.append(float(pnl.loc[xtick].values))
        else:
            xticks_year.append('')
            
        if i == len(xticks) - 1:
            anuual_return.append(float(pnl.loc[xtick].values))
    
    for i, profit in enumerate(anuual_return):
        if i == 0:
            anuual_return_rate.append(float(profit/1e7))
        else:
            anuual_return_rate.append(float((profit-anuual_return[i-1])/(anuual_return[i-1] + investment) ))
    anuual_return_rate = anuual_return_rate[1:]
    
    string_roi = ''
    for i in range(len(anuual_return_rate)):
        if i < len(anuual_return_rate) - 1:
            string_roi += '\n{}-{}: {}'.format(added_year[i], added_year[i+1], np.round(anuual_return_rate[i]*100, 2))
        else:
            string_roi += '\n{}: {}'.format(added_year[i], np.round(anuual_return_rate[i]*100, 2))  
    
    
    # 绘制综合收益曲线
    plt.figure(figsize=(5*len(xticks)//1500 + 10, 5*len(xticks)//1500 + 6,))
    plt.title(f'Porfolio Profit and Loss (PnL) Over Time. \
        \n Investment: {investment} \
        \n Annualized RoI: {string_roi} %' )
        # Sharpe Ratio: {round(100*sharpe_ratio, 4)} %')

    plt.plot(xticks, pnl.values, label='Portfolio PnL', color='blue')
    # plt.plot(xticks, pnl_wo_svc.values, label='Portfolio PnL without service charges', color='orange')
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.xticks(xticks_yearmonth, rotation=80, size=6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    
    # 保存图表
    if not os.path.exists(f'./figures/{foldername}'):
        os.makedirs(f'./figures/{foldername}')
    plt.savefig(f'./figures/{foldername}/Portfolio_PNL.png')
    
    
    # 可视化现金持有
    plt.figure(figsize=(5*len(xticks)//1500 + 10, 5*len(xticks)//1500 + 10,))
    plt.title(f'Currency held overtime. \
        \n Investment: {investment}')
    plt.plot(xticks, investment-results['actual_expenditure'].sum(axis=1).cumsum())
    plt.xlabel('Date')
    plt.ylabel('Currency Held')
    plt.xticks(xticks_yearmonth, rotation=80, size=6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(f'./figures/{foldername}/Portfolio_Held_Currency.png')
    
    
    
    # 可视化持仓情况
    # import pickle
    # pickle.dump(xticks_yearmonth, open('./xticks_yearmonth.pkl', 'wb'))
    # import pdb; pdb.set_trace()
    
    # plt.figure(figsize=(5*len(xticks)//1500 + 10, 5*len(xticks)//1500 + 10,))
    # plt.title(f'Stock position overtime. \
    #     \n Investment: {investment}')
    # bought_amount.plot.area()
    # plt.title('Cumulative Stock Holdings Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Stock Held')
    # # plt.xticks(xticks_yearmonth, rotation=80, size=6)
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.savefig(f'./figures/{foldername}/Stock_Holdings.png')
    
    
    
    
    visualize_single_stock_prices_and_signals(xticks_yearmonth, stock_data, trade_signals, pnl_per_stock, portfolio, 
                                              sma_win_size, indices_to_vis, foldername)
    
    


def visualize_with_index(stock_data, index_data, results, foldername='', sma_win_size=[], investment=1e7):
    # 读取，预处理
    expr = results['expr']
    portfolio = results['total_portfolios']
    trade_signals = results['trade_signals']
    pnl = results['pnl']
    pnl_per_stock = results['pnl_per_stock']
    # bought_amount = results['bought_amount']
    indices_to_vis = np.where((trade_signals*portfolio != 0).any(axis=0).values)[0]
    xticks = np.asarray(trade_signals.index, dtype=str)
    
    # 设置x轴标签
    xticks_yearmonth = []
    xticks_year = []
    added_yearmonth = []
    added_year = []
    anuual_return = []
    anuual_return_rate = []
    for i, xtick in enumerate(xticks):
        yearmonth = xtick[:6]
        if yearmonth not in added_yearmonth:
            added_yearmonth.append(yearmonth)
            xticks_yearmonth.append(xtick)
        else:
            xticks_yearmonth.append('')
        
        year = xtick[:4]
        if year not in added_year: 
            added_year.append(year)
            xticks_year.append(xtick)
            anuual_return.append(float(pnl.loc[xtick].values))
        else:
            xticks_year.append('')
            
        if i == len(xticks) - 1:
            anuual_return.append(float(pnl.loc[xtick].values))
    
    for i, profit in enumerate(anuual_return):
        if i == 0:
            anuual_return_rate.append(float(profit/1e7))
        else:
            anuual_return_rate.append(float((profit-anuual_return[i-1])/(anuual_return[i-1] + investment) ))
    anuual_return_rate = anuual_return_rate[1:]
    
    string_roi = ''
    for i in range(len(anuual_return_rate)):
        if i < len(anuual_return_rate) - 1:
            string_roi += '\n{}-{}: {}'.format(added_year[i], added_year[i+1], np.round(anuual_return_rate[i]*100, 2))
        else:
            string_roi += '\n{}: {}'.format(added_year[i], np.round(anuual_return_rate[i]*100, 2))  
    
    
    # 绘制综合收益曲线
    plt.figure(figsize=(5*len(xticks)//1500 + 10, 5*len(xticks)//1500 + 6,))
        # Sharpe Ratio: {round(100*sharpe_ratio, 4)} %')

    # import pdb; pdb.set_trace()
    ax1 = plt.gca()
    plt.xlabel('Date')
    plt.ylabel('PnL')
    pnl = pnl[0] 
    pnl = pnl / investment
    index_pnl = pd.Series((index_data['Close'] / index_data['Close'].iloc[0]) - 1)
    index_pnl.index = pnl.index
    excess_return = pnl.values - index_pnl.values
    
    # import pdb; pdb.set_trace()
    plt.plot(xticks, pnl.values, label=f'Factor {expr} \'s PnL', color='blue', linewidth=1, zorder=2)
    plt.plot(xticks, index_pnl.values, label='1A0001 PnL', color='green', linewidth=1, zorder=2)
    plt.plot(xticks, excess_return, label='Excess return', color='red', linewidth=2, zorder=2)
    plt.axhline(y=0, color='orange', linestyle='--')
    # import pdb; pdb.set_trace()
    ceilling = np.max([index_pnl.max(), pnl.max()])
    floor = np.min([index_pnl.min(), pnl.min()])
    bound = np.max([np.abs(ceilling), np.abs(floor)])
    plt.ylim(-1.2*bound, 1.2*bound)
    
    
    
    excess_return = pd.Series(excess_return, index=pnl.index).pct_change()
    # sign = np.sign(excess_return.values)
    # win = (sign == 1).sum()
    # day_win_rate = np.round(win/len(sign)*100, 2)
    ym_excess_return = []
    
    # 月度超额收益率
    for i, ym in enumerate(xticks_yearmonth):
        if i == 0:
            ym_excess_return.append(0)
            last_ym = ym
            continue
        
        if ym != '':
            # import pdb; pdb.set_trace()
            ym_excess_return.append(
                                   ((pnl.loc[ym] - pnl.loc[last_ym]) / (1+pnl.loc[last_ym])) - \
                                   ((index_pnl.loc[ym] - index_pnl.loc[last_ym]) / (1+index_pnl.loc[last_ym]))
                                   )
            last_ym = ym
        else:
            ym_excess_return.append(0)
        
    
    ym_excess_return = np.array(ym_excess_return)
    sign = np.sign(ym_excess_return[np.where(ym_excess_return != 0)[0]])
    win = (sign == 1).sum()
    monthly_win_rate = np.round(win/len(sign)*100, 2)
    
    # 年度夏普率 & 最大回撤
    y_info_ratio = {}
    y_max_drawdown = {}
    for i, ym in enumerate(xticks_year):
        if i == 0:
            last_ym = ym
            continue
        if ym != '':
            y_info_ratio.update({
                f'{last_ym[:4]}-{ym[:4]}': calculate_info_ratio(excess_return=pnl.loc[last_ym: ym] - index_pnl.loc[last_ym: ym])
            })
            y_max_drawdown.update({
                f'{last_ym[:4]}-{ym[:4]}': calculate_max_drawdown(pnl.loc[last_ym: ym])
            })
            last_ym = ym
    
    # 换手率
    openprice = pd.DataFrame(data={stock: data['Open PostAdj'] for stock, data in stock_data.items()}) # 买入价
    turnover = ((trade_signals.abs() * openprice).sum(axis=1) / (portfolio * openprice).sum(axis=1)).mean(axis=0)

        
    # import pdb; pdb.set_trace()
    plt.xticks(xticks_yearmonth, rotation=80, size=6)
    plt.twinx()
    ax2 = plt.gca()
    # import pdb; pdb.set_trace()
    nonzero_idx = np.where(ym_excess_return!=0)[0]
    width = nonzero_idx - np.concatenate([np.zeros([1]), nonzero_idx[:-1]], 0)
    width_fulllength = ym_excess_return.copy()
    width_fulllength[np.where(width_fulllength!=0)[0]] = -width
    plt.bar(xticks, ym_excess_return, width=width_fulllength, alpha=0.5, align='edge', label=f'Monthly excess return rate, win rate = {monthly_win_rate} %', zorder=1)
    
    plt.ylim(-2*np.max(np.abs(ym_excess_return)), 2*np.max(np.abs(ym_excess_return)))
    plt.ylabel('Monthly Excess Return Rate')
    plt.axhline(y=0, color='cyan', linestyle='--')
    
    table = '              InfoRatio   MaxDrawdown'
    for k in y_info_ratio.keys():
        table = table + f'\n| {k} |  {y_info_ratio[k].round(4)}  |  {y_max_drawdown[k].round(4)} '
        
    
    # import pdb; pdb.set_trace()
        # \n Investment: {investment} \
        # \n Annualized RoI: {string_roi} % \
        
    plt.title(f'Porfolio Profit and Loss (PnL) Over Time. \
        \n {table}'
        )
    # plt.plot(xticks, pnl_wo_svc.values, label='Portfolio PnL without service charges', color='orange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 保存图表
    if not os.path.exists(f'./figures/{foldername}'):
        os.makedirs(f'./figures/{foldername}')
    plt.savefig(f'./figures/{foldername}/Portfolio_PNL.png')
    
    # import pdb; pdb.set_trace()
    results_to_save = {
        'expr': results['expr'],
        'pnl': pnl.values.tolist(),
        'excess_return': excess_return.tolist(),
        'date': xticks.tolist(),
        'investment': investment,
        'annualized_roi': excess_return[-1] / (len(xticks)//240),
        'turnover': turnover,
        'monthly_win_rate': monthly_win_rate,
    }
    with open(f'./figures/{foldername}/results.json', 'w') as file:
        json.dump(results_to_save, file, indent=4)
    
    
    
    # 可视化现金持有
    plt.figure(figsize=(5*len(xticks)//1500 + 10, 5*len(xticks)//1500 + 10,))
    plt.title(f'Currency held overtime. \
        \n Investment: {investment}')
    plt.plot(xticks, investment-results['actual_expenditure'].sum(axis=1).cumsum())
    plt.xlabel('Date')
    plt.ylabel('Currency Held')
    plt.xticks(xticks_yearmonth, rotation=80, size=6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(f'./figures/{foldername}/Portfolio_Held_Currency.png')
    
    
    
    # 可视化持仓情况
    # import pickle
    # pickle.dump(xticks_yearmonth, open('./xticks_yearmonth.pkl', 'wb'))
    # import pdb; pdb.set_trace()
    
    # plt.figure(figsize=(5*len(xticks)//1500 + 10, 5*len(xticks)//1500 + 10,))
    # plt.title(f'Stock position overtime. \
    #     \n Investment: {investment}')
    # bought_amount.plot.area()
    # plt.title('Cumulative Stock Holdings Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Stock Held')
    # # plt.xticks(xticks_yearmonth, rotation=80, size=6)
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.savefig(f'./figures/{foldername}/Stock_Holdings.png')
    
    # exit(0)
    
    
    visualize_single_stock_prices_and_signals(xticks_yearmonth, stock_data, trade_signals, pnl_per_stock, portfolio, 
                                              sma_win_size, indices_to_vis, foldername)
    


def draw_figures(eval_results: dict=None, figure_path: str=None, vis_each_stock: bool=False) -> None:
    xticks = eval_results['date'] # [d.split('T00')[0] for d in 
    xticks_yearmonth = eval_results['xticks_yearmonth']
    expr = eval_results['expr']
    if len(expr) > 50:
        expr = expr[:4*len(expr)//9] + '\n' + expr[4*len(expr)//9:]
    pnl = eval_results['roi']
    index_pnl = eval_results['index_roi']
    excess_return = eval_results['excess_return']
    ym_excess_return = eval_results['ym_excess_return']
    barwidth_fulllength = eval_results['barwidth_fulllength']
    monthly_win_rate = eval_results['monthly_win_rate']
    index_name = eval_results['stock_pool']
    
    # y_pnl = eval_results['y_roi']
    # y_info_ratio = eval_results['y_info_ratio']
    # y_max_drawdown = eval_results['y_max_drawdown']
    # y_trading_win_rate = eval_results['y_trading_win_rate']
    # years = eval_results['years']
    
    averaged_annualized_roi = eval_results['averaged_annualized_roi']
    turnover = eval_results['turnover']
    overall_trading_win_rate = eval_results['overall_trading_win_rate']
    win_to_loss_ratio = eval_results['win_to_loss_ratio']
    max_drawdown = eval_results['max_drawdown']
    algorithm_volatility = eval_results['algorithm_volatility']
    info_ratio = eval_results['info_ratio']
    alpha = eval_results['alpha']
    beta = eval_results['beta']
    
    # 绘制综合收益曲线
    plt.rcParams['font.family'] = 'SimHei'
    plt.figure(figsize=(6*len(xticks)//1500 + 10, 6*len(xticks)//1500 + 7,))
    # xticks = pd.to_datetime(xticks)
    
    ax1 = plt.gca()
    plt.xlabel(f'日期')
    plt.ylabel(f'收益率 (%)')
    # 日频
    plt.plot(xticks, pnl, label=f'因子收益率 \"{expr}\"', color='blue', linewidth=0.8, zorder=2)
    plt.plot(xticks, index_pnl, label=f'{index_name}收益率', color='green', linewidth=1, zorder=2)
    plt.plot(xticks, excess_return, label='超额收益率', color='red', linewidth=2, zorder=2)
    plt.axhline(y=0, color='orange', linestyle='--')    
    ceilling = np.max([max(index_pnl), max(pnl), max(excess_return)])
    floor = np.min([min(index_pnl), min(pnl), min(excess_return)])
    bound = np.max([np.abs(ceilling), np.abs(floor)])
    plt.ylim(-1.2*bound, 1.2*bound)
    # 设置ax1的xticks
    xticks = np.array(xticks)
    xticks_yearmonth = np.array(xticks_yearmonth)
    firstday_permonth = np.where(xticks_yearmonth != '')[0]
    ax1.set_xticks(xticks[firstday_permonth])  # 设置刻度位置
    ax1.tick_params(axis='x', labelsize=10, rotation=80)
    ax1.set_xticklabels(xticks_yearmonth[firstday_permonth])  # 设置稀疏的刻度标签
    
    plt.twinx()
    ax2 = plt.gca()
    plt.bar(xticks, ym_excess_return, width=np.array(barwidth_fulllength), alpha=0.5, align='edge', label=f'月度超额收益, 月胜率: {monthly_win_rate}', zorder=1)
    plt.ylim(-2*np.max(np.abs(ym_excess_return)), 2*np.max(np.abs(ym_excess_return)))
    plt.ylabel('月度超额收益率')
    plt.axhline(y=0, color='cyan', linestyle='--')
    
    # table = '             |    收益    |   信息比率   |   最大回撤   |     胜率     |'
    # template = '\n {yr:<10}  |   {rt:<6.2f}%  |    {ir:<6.2f}%   |    {md:<6.2f}%   |    {wr:<6.2f}%   |'

    # for yr, rt, ir, md, wr in zip(years, y_pnl, y_info_ratio, y_max_drawdown, y_trading_win_rate):
    #     table += template.format(yr=yr, rt=rt*100, ir=ir, md=md*100, wr=wr*100)
    table = ''
    table += f'\n |  换手率: {turnover*100:.2f}%  |  总胜率: {overall_trading_win_rate*100:.2f}%   |  平均年化: {averaged_annualized_roi*100:.2f}%  |  盈亏比: {win_to_loss_ratio}  |'
    table += f'\n |  最大回撤  {max_drawdown*100:.2f}%  |  波动率: {algorithm_volatility:.6f}  |  Alpha: {alpha:.4f}  |  Beta: {beta:.4f}  | 信息比率: {info_ratio: .4f}'
        
    plt.title(f'{table}', 
        fontsize=10,
        loc='left'
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 保存图表到统一管理的路径
    if figure_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {figure_path}")
    else:
        # 兼容旧的调用方式
        if not os.path.exists(f'./git_ignore_folder/figures'):
            os.makedirs(f'./git_ignore_folder/figures')
        default_path = f'./git_ignore_folder/figures/backtest_result.png'
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {default_path}")