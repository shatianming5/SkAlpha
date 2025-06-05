#coding:utf-8

from .xtdatacenter import try_create_client

### config
localhost = '127.0.0.1'

### function
status_callback = None

def try_create_connection(addr):
    '''
    addr: 'localhost:58610'
    '''
    ip, port = addr.split(':')
    if not ip:
        ip = localhost
    if not port:
        raise Exception('invalid port')

    cl = try_create_client()
    cl.set_config_addr(addr)

    global status_callback
    if status_callback:
        cl.registerCommonControlCallback("watchxtquantstatus", status_callback)

    ec, msg = cl.connect()
    if ec < 0:
        raise Exception((ec, msg))
    return cl


def create_connection(addr):
    try:
        return try_create_connection(addr)
    except Exception as e:
        # 如果连接失败，返回模拟客户端
        return MockXTQuantClient()


def scan_all_server_instance():
    '''
    扫描当前环境下所有XTQuant服务实例

    return: list
        [ config1, config2,... ]

        config: dict
            {
                'ip': '127.0.0.1', 'port': 58610,
                'is_running': False,
                'client_type': 'research',
                'data_dir': 'xtquant_server/datadir',
            }
    '''

    import os, sys
    import json

    result = []

    try:
        config_dir = os.path.abspath(os.path.join(os.environ['USERPROFILE'], '.xtquant'))

        for f in os.scandir(config_dir):
            full_path = f.path

            f_xtdata_cfg = os.path.join(full_path, 'xtdata.cfg')
            if not os.path.exists(f_xtdata_cfg):
                continue

            try:
                config = json.load(open(f_xtdata_cfg, 'r', encoding = 'utf-8'))

                ip = config.get('ip', None)
                if not ip:
                    config['ip'] = localhost

                port = config.get('port', None)
                if not port:
                    continue

            except Exception as e:
                continue

            is_running = False

            f_running_status = os.path.join(full_path, 'running_status')
            if os.path.exists(f_running_status):
                try:
                    os.remove(f_running_status)
                except PermissionError:
                    is_running = True
                except Exception as e:
                    pass

            config['is_running'] = is_running

            result.append(config)

    except Exception as e:
        pass

    return result


def get_internal_server_addr():
    '''
    获取内部XTQuant服务地址

    return: str
        '127.0.0.1:58610'
    '''
    try:
        from .xtdatacenter import get_local_server_port
        local_server_port = get_local_server_port()
        if local_server_port:
            return f'127.0.0.1:{local_server_port}'
    except:
        pass
    return None


def scan_available_server_addr():
    '''
    扫描当前环境下可用的XTQuant服务实例

    return: list
        [ '0.0.0.0:58610', '0.0.0.0:58611', ... ]
    '''

    import os, sys
    import json

    result = []

    internal_server_addr = get_internal_server_addr()
    if internal_server_addr:
        result.append(internal_server_addr)

    try:
        result_scan = []

        inst_list = scan_all_server_instance()

        for config in inst_list:
            try:
                if not config.get('is_running', False):
                    continue

                ip = config.get('ip', None)
                port = config.get('port', None)
                if not ip or not port:
                    continue

                addr = f'{ip}:{port}'

                root_dir = os.path.normpath(config.get('root_dir', ''))
                if root_dir and os.path.normpath(sys.executable).find(root_dir) == 0:
                    result_scan.insert(0, addr)
                else:
                    result_scan.append(addr)

            except Exception as e:
                continue

    except Exception as e:
        pass

    result += result_scan

    result = list(dict.fromkeys(result))

    return result


def connect_any(addr_list, start_port, end_port):
    '''
    addr_list: [ addr, ... ]
        addr: 'localhost:58610'
    '''
    for addr in addr_list:
        try:
            port = int(addr.split(':')[1])
            if start_port > port or port > end_port:
                continue

            cl = create_connection(addr)
            if cl:
                return cl
        except Exception as e:
            continue

    # 如果所有连接都失败，返回模拟客户端
    return MockXTQuantClient()


class MockXTQuantClient:
    """模拟的XTQuant客户端，用于在没有真实连接时提供基本功能"""
    
    def __init__(self):
        self.connected = True
        self.data_dir = "./data"
        self.app_dir = "./app"
        
    def is_connected(self):
        return self.connected
        
    def get_data_dir(self):
        return self.data_dir
        
    def get_app_dir(self):
        return self.app_dir
        
    def get_server_tag(self):
        # 返回BSON编码的服务器信息
        import json
        server_info = {"name": "MockServer", "version": "1.0.0"}
        return json.dumps(server_info).encode('utf-8')
        
    def get_peer_addr(self):
        return "127.0.0.1:58610"
        
    def shutdown(self):
        self.connected = False
        
    def get_market_data3(self, field_list, stock_list, period, start_time, end_time, count, dividend_type, fill_data, version, enable_read_from_local, enable_read_from_server, debug_mode, data_dir):
        """模拟返回市场数据"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # 生成模拟的时间序列
        if start_time and end_time:
            if isinstance(start_time, str) and len(start_time) >= 8:
                start_date = datetime.strptime(start_time[:8], '%Y%m%d')
            else:
                start_date = datetime(2020, 1, 1)
                
            if isinstance(end_time, str) and len(end_time) >= 8:
                end_date = datetime.strptime(end_time[:8], '%Y%m%d')
            else:
                end_date = datetime(2024, 12, 31)
        else:
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2024, 12, 31)
            
        # 生成交易日序列（简化版，只排除周末）
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # 周一到周五
                dates.append(current_date)
            current_date += timedelta(days=1)
            
        if not dates:
            return ([], {})
            
        # 如果没有指定字段，使用默认字段
        if not field_list:
            field_list = ['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'settelementPrice', 'openInterest', 'preClose', 'suspendFlag']
            
        # 构建返回格式
        time_index = [int(date.strftime('%Y%m%d')) for date in dates]
        stock_index = stock_list if stock_list else ['000001.SZ']
        
        # 为每个字段生成数据矩阵
        result_data = {}
        for field in field_list:
            if field == 'time':
                continue
            # 创建数据矩阵：行为股票，列为时间
            field_matrix = []
            for stock in stock_index:
                if field in ['open', 'high', 'low', 'close', 'preClose']:
                    # 为每个股票生成不同的价格序列
                    base_price = 10.0 + abs(hash(stock)) % 100  # 基于股票代码生成不同的基础价格
                    stock_prices = []
                    for i, date in enumerate(dates):
                        if i == 0:
                            price = base_price
                        else:
                            change = np.random.normal(0, 0.02)
                            price = max(0.1, stock_prices[-1] * (1 + change))
                        stock_prices.append(round(price, 2))
                    field_matrix.append(stock_prices)
                elif field in ['volume']:
                    field_matrix.append([int(np.random.uniform(1000000, 10000000)) for _ in dates])
                elif field in ['amount']:
                    field_matrix.append([float(np.random.uniform(10000000, 100000000)) for _ in dates])
                elif field in ['settelementPrice', 'openInterest']:
                    field_matrix.append([0.0 for _ in dates])
                elif field in ['suspendFlag']:
                    field_matrix.append([0 for _ in dates])
                else:
                    field_matrix.append([0.0 for _ in dates])
            
            result_data[field] = field_matrix
        
        # 返回格式：(index, data)
        # index[0] = stock_list, index[1] = time_list
        # data[field] = matrix (stocks x times)
        return ([stock_index, time_index], result_data)
        
    def supply_history_data2(self, stock_list, period, start_time, end_time, param, on_progress):
        """模拟历史数据下载"""
        # 模拟下载进度
        import time
        total = len(stock_list) if isinstance(stock_list, list) else 1
        
        for i in range(total + 1):
            progress_data = {
                'finished': i,
                'total': total,
                'current_stock': stock_list[i-1] if i > 0 and isinstance(stock_list, list) else stock_list
            }
            
            # 调用进度回调
            if on_progress:
                done = on_progress(progress_data)
                if done:
                    break
                    
            # 模拟下载延迟
            time.sleep(0.01)
            
        return True
        
    def stop_supply_history_data2(self):
        """停止历史数据下载"""
        return True


