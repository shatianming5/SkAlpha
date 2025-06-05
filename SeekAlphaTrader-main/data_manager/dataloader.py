'''
Module for data processing.
'''
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Text, List, Dict, Tuple
import json
import baostock as bs
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
import pdb
import subprocess
from xtquant.xtdata import get_market_data, download_history_data2
from xtquant import xtconstant
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_data = {}
        self.test_data = {}
        self.all_data = {}

    # def load_data(self, stock_list='CSI1000', train_start_time=20150000, train_end_time=20200000, test_start_time=20200000, test_end_time=20231100,):
    #     if isinstance(stock_list, str):
    #         import pdb; pdb.set_trace()
    #         try:
    #             from stock_list import CSI_list
    #             stock_list_all = CSI_list[stock_list]
    #             stock_list_all = [code + '.csv' for code in stock_list_all]
    #         except:
    #             stock_list_all = os.listdir(self.data_dir)
    #             stock_list_all.sort()
    #     elif isinstance(stock_list, list):
    #         stock_list_all = [code + '.csv' for code in stock_list]
    #     else:
    #         raise NotImplementedError

    #     trade_amounts = []
    #     stock_codes = []
    #     for i, filename in enumerate(stock_list_all):
    #         if filename.endswith('.csv'):
    #             stock_code = os.path.splitext(filename)[0]
    #             file_path = os.path.join(self.data_dir, filename)
    #             try:
    #                 df = pd.read_csv(file_path, header=None)
    #             except:
    #                 continue
    #             df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Circulating MarketCap', 'Turnover', 'Total MarketCap']

    #             # 忽略退市股票: 为了防止数据窥测，仅按照训练时间（已知）筛选股票
    #             if len(df[df['Date'] < train_start_time]) <= 0 or len(df[df['Date'] > train_end_time]) <= 0:
    #                 continue

    #             # 记录一年内的日均成交额
    #             trade_amount = df['Amount'][(train_end_time - 10000 < df['Date']) * (df['Date'] < train_end_time)].mean()
    #             trade_amounts.append(trade_amount)
    #             stock_codes.append(stock_code)

    #             self.train_data[stock_code] = df.loc[(train_start_time <= df['Date']) & (df['Date'] <= train_end_time), ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount']]
    #             self.test_data[stock_code] = df.loc[(test_start_time <= df['Date']) & (df['Date'] <= test_end_time), ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount']]

    #             print(f"Stock {stock_code} loaded. Training data has {len(self.train_data[stock_code])} days and testing {len(self.test_data[stock_code])} days")

    #     # 选沪深XX股
    #     if isinstance(stock_list, str) and stock_list.startswith('CSI'):
    #         num_stocks = int(stock_list.split('CSI')[-1])
    #         bottom_stock_codes = np.asarray(trade_amounts).argsort()[num_stocks:]
    #         bottom_stock_codes = np.asarray(stock_codes)[bottom_stock_codes]

    #         # 剔除交易额排名靠后的股票
    #         for c in bottom_stock_codes:
    #             self.train_data.pop(c, None)
    #             self.test_data.pop(c, None)

    #     # import pdb; pdb.set_trace()
    #     print(f"{len(self.train_data)} stocks are loaded.")
    #     self.clean_data()
    #     self.unify_data()
    #     return self.train_data, self.test_data

    def load_all_data(self, train_start_time, train_end_time, test_start_time, test_end_time):
        # 确认数据
        stock_list_all_unadjusted = os.listdir(os.path.join(self.data_dir, 'pricedata_0_csv'))
        stock_list_all_unadjusted.sort()
        stock_list_all_postadj = os.listdir(os.path.join(self.data_dir, 'pricedata_adjusted_csv'))
        stock_list_all_postadj.sort()
        for stock in stock_list_all_postadj:
            assert stock in stock_list_all_unadjusted

        # import pdb; pdb.set_trace()
        # assert stock_list_all_unadjusted == stock_list_all_postadjusted

        for i, filename in enumerate(stock_list_all_postadj):
            if filename.endswith('.csv'):
                stock_code = os.path.splitext(filename)[0]
                # if 'SZ000760' in stock_code:
                #     import pdb; pdb.set_trace()
                # 读取未复权数据
                try:
                    file_path_postadj = os.path.join(self.data_dir, 'pricedata_adjusted_csv', filename)
                    df_postadj = pd.read_csv(file_path_postadj, header=None)
                except:
                    continue

                df_postadj.columns = ['Date', 'Open PostAdj', 'High PostAdj', 'Low PostAdj', 'Close PostAdj', 'Volume PostAdj', 'Circulating MarketCap', 'Turnover PostAdj', 'Total MarketCap']
                # 去除截至训练结束时间已经退市的股票
                if len(df_postadj[df_postadj['Date'] > train_end_time]) <= 0:
                    continue

                # 读取后复权数据

                try:
                    file_path_unadj = os.path.join(self.data_dir, 'pricedata_0_csv', filename)
                    df_unadj = pd.read_csv(file_path_unadj, header=None)
                except:
                    continue

                df_unadj.columns = ['Date', 'Open UnAdj', 'High UnAdj', 'Low UnAdj', 'Close UnAdj', 'Volume UnAdj', 'Ceiling', 'Floor', 'Turnover Rate']
                # 去除截至训练结束时间已经退市的股票
                if len(df_unadj[df_unadj['Date'] > train_end_time]) <= 0:
                    continue

                assert (df_unadj['Date'] == df_postadj['Date']).all()

                # df_postadj.ffill(inplace=True)
                # df_unadj.ffill(inplace=True)

                df = pd.concat([df_unadj, df_postadj.iloc[:, 1:]], axis=1)
                # import pdb; pdb.set_trace()

                self.all_data[stock_code] = df.loc[(train_start_time <= df['Date']) & (df['Date'] <= test_end_time)]

                # 数据加载进度信息已在主程序中统一管理
                pass

        # 数据加载完成信息已在主程序中统一输出
        all_dates = sorted(set().union(*[data['Date'] for data in self.all_data.values()]))
        for stock, data in self.all_data.items():
            data.set_index('Date', inplace=True)
            self.all_data[stock] = data.reindex(all_dates)

        # import pdb; pdb.set_trace()
        return self.all_data

    def clean_data(self):
        assert self.train_data != {}
        for stock_code, df in self.train_data.items():
            df.ffill(inplace=True)
        for stock_code, df in self.test_data.items():
            df.ffill(inplace=True)
        return 0

    def unify_data(self):
        # 对齐所有股票数据的日期，通过前向填充缺失数据
        # TODO: 确认合理性
        all_dates = sorted(set().union(*[data['Date'] for data in self.train_data.values()]))
        for stock, data in self.train_data.items():
            data.set_index('Date', inplace=True)
            self.train_data[stock] = data.reindex(all_dates, method='ffill')

        all_dates = sorted(set().union(*[data['Date'] for data in self.test_data.values()]))
        for stock, data in self.test_data.items():
            data.set_index('Date', inplace=True)
            self.test_data[stock] = data.reindex(all_dates, method='ffill')




class BaoStockLoader:
    """
    适合策略开发，增量数据处理逻辑暂未开发
    """
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        lg = bs.login()
        if lg.error_code != '0':
            print("登录失败，错误信息：" + lg.error_msg)

        # 登出Baostock账号
        # bs.logout()

    def transform_date_format(self, date):
        ymd = date.split('-')
        return int(''.join(ymd))

    def transformer_to_float(self, data):
        return float(data)

    def load_stock_data(self, code="sh.000001", start_date='2020-01-01', end_date='2020-12-31', get_unadj=False):
        # 将股票代码转换为baostock格式
        if code.endswith('.SH') or code.endswith('.SZ'):
            code = code.split('.')[1].lower() + '.' + code.split('.')[0]

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')

        file_name = f'{code}_{start_date}_{end_date}{"_with_unadj" if get_unadj else ""}.csv'
        os.makedirs(f'{self.data_dir}/pv_data/', exist_ok=True)
        # 如果数据已经存在，则直接读取
        if file_name in os.listdir(f'{self.data_dir}/pv_data'):
            # print(f'{self.data_dir}/{code}_{start_date}_{end_date}.csv loaded.')
            data = pd.read_csv(f'{self.data_dir}/pv_data/{file_name}', index_col='date')
            # 数据文件加载信息已在主程序中统一管理
            return data
        else:
            # 获取个股数据
            rs = bs.query_history_k_data_plus(code, "date,open,high,low,close,preclose,volume,amount,turn", # ,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST
                                            start_date=start_date, end_date=end_date, frequency='d', adjustflag='2')
            if rs.error_code != '0':
                print("请求A股指数数据失败，错误信息：" + rs.error_msg)

            data = rs.get_data()
            data = data.set_index('date')


            if get_unadj:
                # 获取不复权数据
                rs = bs.query_history_k_data_plus(code, "date,open,close", start_date=start_date, end_date=end_date, frequency='d', adjustflag='3')
                # 重命名
                if rs.error_code != '0':
                    print("请求A股指数数据失败，错误信息：" + rs.error_msg)
                data_unadj = rs.get_data()
                data_unadj = data_unadj.set_index('date').replace('', np.nan).astype(float)
                data_unadj.columns = ['open_unadj', 'close_unadj']

                # 合并数据
                data = pd.concat([data, data_unadj], axis=1)

            data = data.apply(lambda x: pd.to_numeric(x, errors='coerce'))
            os.makedirs(os.path.dirname(f'{self.data_dir}/pv_data/{file_name}'), exist_ok=True)
            data.to_csv(f'{self.data_dir}/pv_data/{file_name}')
            # 数据文件保存信息已在主程序中统一管理
            return data

    def load_hsi300(self, date=None):
        if date is not None:
            date = str(date)
            date = '-'.join([date[:4], date[4:6], date[6:8]])

        # 获取沪深300成分股
        rs = bs.query_hs300_stocks(date=date)
        if rs.error_code != '0':
            print('请求A股沪深300数据失败，错误信息：' + rs.error_msg)

        # 打印结果集
        hs300_stocks = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            hs300_stocks.append(rs.get_row_data())
        # import pdb; pdb.set_trace()
        result = pd.DataFrame(hs300_stocks, columns=rs.fields)

        return result #.code.apply(lambda x: x.replace('sh.', 'SH').replace('sz.', 'SZ')).values

    def load_zz500_stocks(self, date=""):
        
        rs = bs.query_zz500_stocks(date=date)
        # 查询错误信息已在主程序中统一处理
        if rs.error_code != '0':
            raise Exception(f'查询中证500成分股失败，错误信息：{rs.error_msg}')

        # 打印结果集
        zz500_stocks = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            zz500_stocks.append(rs.get_row_data())
        result = pd.DataFrame(zz500_stocks, columns=rs.fields)['code']
        result = result.apply(lambda x: x.split('.')[-1] + '.' + x.split('.')[0].upper())
        return result 

    def load_index_stocklist(self, index_name=None, date=None):
        date = str(date.date())
        # 获取成分股
        if index_name == "沪深300" or index_name == "sh.000300":
            rs = bs.query_hs300_stocks(date=date)
        elif index_name == "中证500" or index_name == "sh.000905":
            rs = bs.query_zz500_stocks(date=date)
        elif index_name == "上证50" or index_name == "sh.000016":
            rs = bs.query_sz50_stocks(date=date)
        else:
            raise NotImplementedError(f"不支持的股票池：{index_name}。")
        
        if rs.error_code != '0':
            raise Exception(f'请求A股指数{index_name}成分股列表失败，错误信息：' + rs.error_msg)

        # 打印结果集
        stocks = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            stocks.append(rs.get_row_data())
            
        result = pd.DataFrame(stocks, columns=rs.fields)['code']
        result = result.apply(lambda x: x.split('.')[-1] + '.' + x.split('.')[0].upper())
        return result
    
    def load_index_stocklist_timerange(self, index_name: str, start_date: datetime, end_date: datetime, use_cache=True):
        json_file = '{}/constituent_stocks/{}_{}_{}.json'.format(self.data_dir, index_name.split('.')[1], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        # 成分股列表加载信息已在主程序中统一管理

        if use_cache and os.path.exists(json_file):
            constituent_stock_codes = json.load(open(json_file))
            constituent_stock_codes = {pd.to_datetime(k): v for k, v in constituent_stock_codes.items()}
            # 成分股列表加载完成信息已在主程序中统一输出
            return constituent_stock_codes
        
        else:
            # 按半年一次获取指数成分股
            dates = []
            current_date = start_date
            while current_date <= end_date:
                if current_date.month in [1, 7] and current_date.day == 1:
                    dates.append(current_date)
                # 移动到下个月
                next_month = current_date + pd.DateOffset(months=1)
                current_date = pd.Timestamp(year=next_month.year, month=next_month.month, day=1)
            dates = [start_date] + dates

            # 获取对应日期的指数成分股
            constituent_stock_dict = {}
            for date in dates:
                stockcodes = self.load_index_stocklist(index_name, date).to_list()
                constituent_stock_dict[date] = stockcodes

            # 缓存
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            with open(json_file, 'w') as file:
                json.dump({k.strftime('%Y-%m-%d'): v for k, v in constituent_stock_dict.items()}, file, indent=4)

            print(f'{index_name}成分股列表下载并缓存完成，共{len(constituent_stock_dict)}个日期，共{len(set(sum(constituent_stock_dict.values(), [])))}个成分股。')
            return constituent_stock_dict

    def load_stock_price_timerange(self, constituent_stock_codes, backtest_start_time, backtest_end_time, use_cache=True):
        print(f'开始加载{backtest_start_time.strftime("%Y%m%d")}至{backtest_end_time.strftime("%Y%m%d")}的行情数据...')
        cache_dir = f'{self.data_dir}/pickle'
        cache_filename = f'stock_data_{backtest_start_time.strftime("%Y%m%d")}_{backtest_end_time.strftime("%Y%m%d")}.pkl'
        cache_path = os.path.join(cache_dir, cache_filename)
        os.makedirs(cache_dir, exist_ok=True)

        # 检查是否存在缓存
        if use_cache and os.path.exists(cache_path):
            print("从缓存加载行情数据...")
            with open(cache_path, 'rb') as f:
                price_data = pickle.load(f)
            return price_data
        
        else:
            # 取所有股票代码并集
            all_stock_codes = set()
            date_list = list(constituent_stock_codes.keys())
            date_list.sort()
            for date in date_list:
                all_stock_codes.update(constituent_stock_codes[date])
            all_stock_codes = list(all_stock_codes)


            # 获取个股进入和退出指数的时间
            stock_info_list = {}
            for code in all_stock_codes:
                start_date = None
                for date in date_list:
                    # 起始日期
                    if code in constituent_stock_codes[date] and start_date is None:
                        start_date = date
                        stock_info_list.update({code: {'start_date': start_date}})
                        # 如果当前日期是最后一个日期，则结束日期为回测结束日期
                        if date == date_list[-1]:
                            stock_info_list[code].update({'end_date': backtest_end_time})
                        continue

                    # 结束日期为当前code不再是成分股的日期
                    if code not in constituent_stock_codes[date] and start_date is not None:
                        stock_info_list[code].update({'end_date': date})
                        break
                    # 如果在更新成分股的最后一个日期，该股还在成分股中，则结束日期为回测结束日期
                    if date == date_list[-1]:
                        stock_info_list[code].update({'end_date': backtest_end_time})
                        break
                
                assert 'end_date' in stock_info_list[code]
                #     pdb.set_trace()
                #     stock_info_list[code].update({'end_date': backtest_end_time})

            

            stock_price_dict = {}
            for stock_code, stock_info in tqdm(stock_info_list.items(), desc="加载股票数据"):
                stock_price_dict[stock_code] = self.load_stock_data(code=stock_code, start_date=stock_info['start_date'], end_date=stock_info['end_date'], get_unadj=True)


            
            data_dict = {}
            for code in all_stock_codes:
                df = stock_price_dict[code].copy()
                df['return'] = df['close'] / df['close'].shift(1) - 1
                # df.drop(['open_unadj', 'close_unadj'], axis=1, inplace=True)
                # 重置索引，将date变成列
                df = df.reset_index()
                # 为每个股票的数据添加code列
                df['instrument'] = code
                data_dict[code] = df

            # 合并所有股票数据
            combined_df = pd.concat(data_dict.values())
            # 设置多级索引并按日期排序
            combined_df.set_index(['date', 'instrument'], inplace=True)
            combined_df = combined_df.groupby(['date', 'instrument']).first()
            # 重命名date列
            combined_df.index = combined_df.index.rename(['datetime', 'instrument'])
            
            # 确保date索引为datetime类型
            combined_df.index = combined_df.index.set_levels([pd.to_datetime(combined_df.index.levels[0]), combined_df.index.levels[1]])
            combined_df = combined_df.sort_index()

            # 保存到缓存
            with open(cache_path, 'wb') as f:
                pickle.dump(combined_df, f)
                
            return combined_df


    def load_stock_industry(self, code="sh.000001", start_date='2020-01-01', end_date='2020-12-31'):
        # 获取行业数据
        rs = bs.query_stock_industry()

        # 打印结果集
        industry_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            industry_list.append(rs.get_row_data())
        result = pd.DataFrame(industry_list, columns=rs.fields)
        result.to_csv(f"{self.data_dir}/stock_industry.csv", encoding="gbk", index=False)
        return result





from xtquant.xtdata import get_market_data, download_history_data2
from xtquant import xtconstant
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
import subprocess

class QMTDataLoader(BaoStockLoader):
    def __init__(self, data_dir='./data', simulate_client=False):
        super().__init__(data_dir)
        load_dotenv()
        self.simulate_client = simulate_client


    def load_stock_price_timerange(self, constituent_stock_codes, backtest_start_time, backtest_end_time, use_cache=True):
        print(f'开始加载{backtest_start_time.strftime("%Y%m%d")}至{backtest_end_time.strftime("%Y%m%d")}的行情数据...')
        cache_dir = f'{self.data_dir}/pickle'
        cache_filename = f'stock_data_{backtest_start_time.strftime("%Y%m%d")}_{backtest_end_time.strftime("%Y%m%d")}.pkl'
        cache_path = os.path.join(cache_dir, cache_filename)
        os.makedirs(cache_dir, exist_ok=True)

        # 检查是否存在缓存
        if use_cache and os.path.exists(cache_path):
            print("从缓存加载行情数据...")
            with open(cache_path, 'rb') as f:
                price_data = pickle.load(f)
            return price_data
        
        else:
            # 取所有股票代码并集
            all_stock_codes = set()
            date_list = list(constituent_stock_codes.keys())
            date_list.sort()
            for date in date_list:
                all_stock_codes.update(constituent_stock_codes[date])
            all_stock_codes = list(all_stock_codes)

            # pdb.set_trace()

            stock_price_dict = self.load_stock_data(code=all_stock_codes, start_date=backtest_start_time, end_date=backtest_end_time, get_unadj=True)

            # 保存到缓存
            with open(cache_path, 'wb') as f:
                pickle.dump(stock_price_dict, f)
            return stock_price_dict


    def load_stock_data(self, code="sh.000001", start_date='2020-01-01', end_date='2020-12-31', get_unadj=False):
        print(f"开始加载{code} {start_date} 至 {end_date} 的行情数据...")
        # if isinstance(start_date, str) and len(start_date) == 10:
        #     start_date = datetime.strptime(start_date, '%Y-%m-%d')
        # if isinstance(end_date, str) and len(end_date) == 10:
        #     end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # 将日期转换为字符串
        if isinstance(start_date, str) and len(start_date) == 10: # 日期格式为2020-01-01
            start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d%H%M%S')
        elif isinstance(start_date, pd.Timestamp): # 日期格式为20200101
            start_date = start_date.strftime('%Y%m%d%H%M%S')
        if isinstance(end_date, str) and len(end_date) == 10: # 日期格式为2020-01-01
            end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d%H%M%S')
        elif isinstance(end_date, pd.Timestamp): # 日期格式为20200101
            end_date = end_date.strftime('%Y%m%d%H%M%S')

        # 将股票代码转换为QMT格式: 600001.SH
        if isinstance(code, str): 
            if code.endswith('.sh') or code.endswith('.sz'):
                code = code.upper()
            elif code.startswith('sh.') or code.startswith('sz.'):
                code = code.split('.')[1] + '.' + code.split('.')[0].upper()
            code_list = [code]
        else:
            code_list = code
        
        # 数据储存目录："D:\国金证券QMT交易端\userdata_mini\datadir\"
        results_no_adjust = None
        results_back_adjust = None
        # if code == '300496.SZ':
        #     import pdb; pdb.set_trace()
        
        count = 0
        try:
            while (results_no_adjust is None or results_no_adjust['time'].size == 0 or results_back_adjust['time'].size == 0) and count <= 10:
                download_history_data2(code_list, period='1d', start_time=start_date, end_time=end_date, callback=None, incrementally=True)
                results_back_adjust = get_market_data(field_list=[], stock_list=code_list, period='1d', start_time=start_date, end_time=end_date, dividend_type='back', fill_data=True)
                results_no_adjust = get_market_data(field_list=['time', 'open', 'close'], stock_list=code_list, period='1d', start_time=start_date, end_time=end_date, dividend_type='none', fill_data=True)
                count += 1
                
        except Exception as e:
            if self.simulate_client:
                qmt_exe_path = os.getenv('SIMULATE_QMT_EXE_PATH')
                qmt_path = os.getenv('SIMULATE_QMT_PATH')
            else:
                qmt_exe_path = os.getenv('QMT_EXE_PATH')
                qmt_path = os.getenv('QMT_PATH')

            # 打开可执行文件XtMiniQmt.exe "D:\\QMT\\"需要根据自己的路径修改
            print("打开QMT交易端")
            process = subprocess.Popen(qmt_exe_path)
            print(qmt_exe_path)
            session_id = 12345
            xt_trader = XtQuantTrader(qmt_path, session_id)
            callback = XtQuantTraderCallback()
            xt_trader.register_callback(callback)
            xt_trader.start()
            connect_result = xt_trader.connect()
            if connect_result != 0:
                print(f'xtquant链接失败，连接结果: {connect_result}')
                print('QMT数据源不可用，将使用缓存数据或返回空结果')
                # 不退出程序，而是抛出异常让上层处理
                raise ConnectionError(f'QMT连接失败，连接结果: {connect_result}')


        if isinstance(code, str):
            df_dict = {}
            # 不复权数据
            for field in results_back_adjust.keys():
                if field != 'time' and isinstance(results_back_adjust[field], pd.DataFrame):
                    df_dict[field] = results_back_adjust[field].loc[code]  # 因为只有一个股票，取第一行

                # 后复权数据
                for field in results_no_adjust.keys():
                    if field in ['open', 'close'] and isinstance(results_no_adjust[field], pd.DataFrame):
                        df_dict[field + '_unadj'] = results_no_adjust[field].loc[code]  # 因为只有一个股票，取第一行

            df = pd.DataFrame(df_dict)
            df.index = pd.to_datetime(df.index.astype(str))
            df.index.name = 'date'
            df = df.sort_index()
            df = df.drop(columns=['settelementPrice', 'openInterest', 'preClose', 'suspendFlag'])

            # 替换0值为NaN
            df = df.replace(0, np.nan)
            return df
        else:
            for field in results_back_adjust.keys():
                if field != 'time' and isinstance(results_back_adjust[field], pd.DataFrame):
                    # 将数据转换为长格式
                    df = results_back_adjust[field].reset_index()
                    df = df.melt(id_vars='index', var_name='datetime', value_name=field)
                    df = df.rename(columns={'index': 'instrument'})
                    
                    if 'stacked_data' not in locals():
                        stacked_data = df
                    else:
                        stacked_data = pd.merge(stacked_data, df, on=['instrument', 'datetime'])

            # 添加不复权数据
            for field in results_no_adjust.keys():
                if field in ['open', 'close'] and isinstance(results_no_adjust[field], pd.DataFrame):
                    df = results_no_adjust[field].reset_index()
                    df = df.melt(id_vars='index', var_name='datetime', value_name=f'{field}_unadj')
                    df = df.rename(columns={'index': 'instrument'})
                    stacked_data = pd.merge(stacked_data, df, on=['instrument', 'datetime'])

            # 转换日期格式并设置多重索引
            stacked_data['datetime'] = pd.to_datetime(stacked_data['datetime'].astype(str))
            stacked_data = stacked_data.set_index(['datetime', 'instrument'])
            stacked_data = stacked_data.sort_index()

            # 替换0值为NaN
            stacked_data = stacked_data.replace(0, np.nan)
            stacked_data = stacked_data.drop(columns=['settelementPrice', 'openInterest', 'preClose', 'suspendFlag'])

            return stacked_data


# if __name__ == '__main__':
#     stock_list = 'CSI1000'
#     num_stocks = 9999999
#     train_start_time = 20100201
#     train_end_time = 20230101
#     test_start_time = "2019-04-18"
#     test_end_time = "2024-04-18"
#     num_stocks_to_hold = 1

#     # dev_data_loader = DataLoader('../datasets/financialdata/pricedata_adjusted_csv/')
#     # dev_stock_data, backtest_stock_data = dev_data_loader.load_data(stock_list=stock_list, 
#     #                                            train_start_time=train_start_time, 
#     #                                            train_end_time=train_end_time, 
#     #                                            test_start_time=test_start_time,
#     #                                            test_end_time=test_end_time)

#     index_loader = BaoStockLoader()
#     index_data = index_loader.load_hsi300(train_end_time)  # load_data(start_date=train_start_time, end_date=train_end_time)
#     index_data = index_loader.load_data(code="sh.000001", start_date=test_start_time, end_date=test_end_time)
#     import pdb; pdb.set_trace()
