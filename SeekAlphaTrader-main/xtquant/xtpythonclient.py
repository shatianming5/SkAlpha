# coding: utf-8
"""
模拟的xtpythonclient模块
用于替代缺失的编译扩展模块
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable


# 请求类
class SubscribeReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class UnsubscribeReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class OrderStockReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_strStockCode = ""
        self.m_nOrderType = 0
        self.m_nOrderVolume = 0
        self.m_nPriceType = 0
        self.m_dPrice = 0.0
        self.m_strStrategyName = ""
        self.m_strOrderRemark = ""
        self.m_strOrderRemarkNew = ""
        self.m_dOrderAmount = 0.0
        self.m_strStockCode1 = ""


class CancelOrderStockReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_nOrderID = 0
        self.m_strMarket = ""
        self.m_nMarket = 0
        self.m_strOrderSysID = ""


class QueryAccountInfosReq:
    def __init__(self):
        pass


class QueryAccountStatusReq:
    def __init__(self):
        pass


class QueryStockAssetReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class QueryStockOrdersReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_bCancelableOnly = False


class QueryStockTradesReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class QueryStockPositionsReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_strStockCode = ""


class QueryCreditDetailReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class QueryStkCompactsReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class QueryCreditSubjectsReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class QueryCreditSloCodeReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class QueryCreditAssureReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class QueryNewPurchaseLimitReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class QueryIPODataReq:
    def __init__(self):
        pass


class TransferParam:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_nTransferDirection = 0
        self.m_dPrice = 0.0
        self.m_strStockCode = ""
        self.m_nVolume = 0
        self.m_nTransferType = 0


class QueryComFundReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class QueryComPositionReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class SmtQueryQuoterReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class SmtNegotiateOrderReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_strSrcGroupID = ""
        self.m_strOrderCode = ""
        self.m_strDate = ""
        self.m_dAmount = 0.0
        self.m_dApplyRate = 0.0


class SmtAppointmentOrderReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_strOrderCode = ""
        self.m_strDate = ""
        self.m_dAmount = 0.0
        self.m_dApplyRate = 0.0


class SmtAppointmentCancelReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_strApplyID = ""


class SmtQueryOrderReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class SmtQueryCompactReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


class SmtCompactRenewalReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_strCashCompactID = ""
        self.m_strOrderCode = ""
        self.m_nDeferDays = 0
        self.m_nDeferNum = 0
        self.m_dApplyRate = 0.0


class SmtCompactReturnReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""
        self.m_strSrcGroupID = ""
        self.m_strCashCompactID = ""
        self.m_strOrderCode = ""
        self.m_dOccurAmount = 0.0


class QueryPositionStatisticsReq:
    def __init__(self):
        self.m_nAccountType = 0
        self.m_strAccountID = ""


# 响应类
class MockResponse:
    def __init__(self):
        self.m_strAccountID = ""
        self.m_nOrderID = 0
        self.m_strStrategyName = ""
        self.m_strOrderRemark = ""
        self.m_strErrorMsg = ""
        self.m_nCancelResult = 0
        self.m_strOrderSysID = ""
        self.m_bSuccess = True
        self.m_strMsg = ""
        self.m_strApplyID = ""


# 主要的异步客户端类
class XtQuantAsyncClient:
    def __init__(self, path: bytes, name: str, session: str):
        self.path = path
        self.name = name
        self.session = session
        self.seq_counter = 0
        self.callbacks = {}
        self.initialized = False
        self.started = False
        self.connected = False
        
    def nextSeq(self) -> int:
        """获取下一个序列号"""
        self.seq_counter += 1
        return self.seq_counter
        
    def setTimeout(self, timeout: int):
        """设置超时时间"""
        pass
        
    def init(self) -> int:
        """初始化客户端"""
        self.initialized = True
        return 0
        
    def start(self) -> int:
        """启动客户端"""
        self.started = True
        return 0
        
    def stop(self) -> int:
        """停止客户端"""
        self.started = False
        return 0
        
    def connect(self) -> int:
        """连接服务器"""
        self.connected = True
        return 0
        
    # 绑定回调方法
    def bindOnSubscribeRespCallback(self, callback):
        self.callbacks['subscribe_resp'] = callback
        
    def bindOnUnsubscribeRespCallback(self, callback):
        self.callbacks['unsubscribe_resp'] = callback
        
    def bindOnQueryStockAssetCallback(self, callback):
        self.callbacks['query_stock_asset'] = callback
        
    def bindOnQueryStockOrdersCallback(self, callback):
        self.callbacks['query_stock_orders'] = callback
        
    def bindOnQueryStockTradesCallback(self, callback):
        self.callbacks['query_stock_trades'] = callback
        
    def bindOnQueryStockPositionsCallback(self, callback):
        self.callbacks['query_stock_positions'] = callback
        
    def bindOnQueryCreditDetailRespCallback(self, callback):
        self.callbacks['query_credit_detail'] = callback
        
    def bindOnQueryStkCompactsRespCallback(self, callback):
        self.callbacks['query_stk_compacts'] = callback
        
    def bindOnQueryCreditSubjectsRespCallback(self, callback):
        self.callbacks['query_credit_subjects'] = callback
        
    def bindOnQueryCreditSloCodeRespCallback(self, callback):
        self.callbacks['query_credit_slo_code'] = callback
        
    def bindOnQueryCreditAssureRespCallback(self, callback):
        self.callbacks['query_credit_assure'] = callback
        
    def bindOnQueryNewPurchaseLimitCallback(self, callback):
        self.callbacks['query_new_purchase_limit'] = callback
        
    def bindOnQueryIPODataCallback(self, callback):
        self.callbacks['query_ipo_data'] = callback
        
    def bindOnTransferRespCallback(self, callback):
        self.callbacks['transfer_resp'] = callback
        
    def bindOnQueryComFundRespCallback(self, callback):
        self.callbacks['query_com_fund'] = callback
        
    def bindOnSmtQueryQuoterRespCallback(self, callback):
        self.callbacks['smt_query_quoter'] = callback
        
    def bindOnSmtQueryOrderRespCallback(self, callback):
        self.callbacks['smt_query_order'] = callback
        
    def bindOnSmtQueryCompactRespCallback(self, callback):
        self.callbacks['smt_query_compact'] = callback
        
    def bindOnQueryPositionStatisticsRespCallback(self, callback):
        self.callbacks['query_position_statistics'] = callback
        
    def bindOnExportDataRespCallback(self, callback):
        self.callbacks['export_data'] = callback
        
    def bindOnSyncTransactionFromExternalRespCallback(self, callback):
        self.callbacks['sync_transaction'] = callback
        
    def bindOnQueryAccountInfosCallback(self, callback):
        self.callbacks['query_account_infos'] = callback
        
    def bindOnQueryAccountStatusCallback(self, callback):
        self.callbacks['query_account_status'] = callback
        
    def bindOnOrderStockRespCallback(self, callback):
        self.callbacks['order_stock_resp'] = callback
        
    def bindOnCancelOrderStockRespCallback(self, callback):
        self.callbacks['cancel_order_stock_resp'] = callback
        
    def bindOnDisconnectedCallback(self, callback):
        self.callbacks['disconnected'] = callback
        
    def bindOnUpdateAccountStatusCallback(self, callback):
        self.callbacks['update_account_status'] = callback
        
    def bindOnStockAssetCallback(self, callback):
        self.callbacks['stock_asset'] = callback
        
    def bindOnStockOrderCallback(self, callback):
        self.callbacks['stock_order'] = callback
        
    def bindOnStockTradeCallback(self, callback):
        self.callbacks['stock_trade'] = callback
        
    def bindOnStockPositionCallback(self, callback):
        self.callbacks['stock_position'] = callback
        
    def bindOnOrderErrorCallback(self, callback):
        self.callbacks['order_error'] = callback
        
    def bindOnCancelErrorCallback(self, callback):
        self.callbacks['cancel_error'] = callback
        
    def bindOnSmtAppointmentRespCallback(self, callback):
        self.callbacks['smt_appointment_resp'] = callback
        
    # 操作方法
    def subscribeWithSeq(self, seq: int, req: SubscribeReq):
        """订阅"""
        # 模拟异步响应
        def _respond():
            time.sleep(0.1)
            if 'subscribe_resp' in self.callbacks:
                self.callbacks['subscribe_resp'](seq, MockResponse())
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    def unsubscribeWithSeq(self, seq: int, req: UnsubscribeReq):
        """取消订阅"""
        def _respond():
            time.sleep(0.1)
            if 'unsubscribe_resp' in self.callbacks:
                self.callbacks['unsubscribe_resp'](seq, MockResponse())
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    def orderStockWithSeq(self, seq: int, req: OrderStockReq):
        """股票下单"""
        def _respond():
            time.sleep(0.1)
            resp = MockResponse()
            resp.m_nOrderID = seq  # 使用seq作为订单ID
            if 'order_stock_resp' in self.callbacks:
                self.callbacks['order_stock_resp'](seq, resp)
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    def cancelOrderStockWithSeq(self, seq: int, req: CancelOrderStockReq):
        """撤销股票订单"""
        def _respond():
            time.sleep(0.1)
            resp = MockResponse()
            resp.m_nCancelResult = 0  # 成功
            resp.m_nOrderID = req.m_nOrderID
            resp.m_strOrderSysID = req.m_strOrderSysID
            if 'cancel_order_stock_resp' in self.callbacks:
                self.callbacks['cancel_order_stock_resp'](seq, resp)
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    def queryAccountInfosWithSeq(self, seq: int, req: QueryAccountInfosReq):
        """查询账户信息"""
        def _respond():
            time.sleep(0.1)
            if 'query_account_infos' in self.callbacks:
                self.callbacks['query_account_infos'](seq, [])
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    def queryAccountStatusWithSeq(self, seq: int, req: QueryAccountStatusReq):
        """查询账户状态"""
        def _respond():
            time.sleep(0.1)
            if 'query_account_status' in self.callbacks:
                self.callbacks['query_account_status'](seq, MockResponse())
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    def queryStockAssetWithSeq(self, seq: int, req: QueryStockAssetReq):
        """查询股票资产"""
        def _respond():
            time.sleep(0.1)
            if 'query_stock_asset' in self.callbacks:
                self.callbacks['query_stock_asset'](seq, MockResponse())
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    def queryStockOrdersWithSeq(self, seq: int, req: QueryStockOrdersReq):
        """查询股票订单"""
        def _respond():
            time.sleep(0.1)
            if 'query_stock_orders' in self.callbacks:
                self.callbacks['query_stock_orders'](seq, [])
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    def queryStockTradesWithSeq(self, seq: int, req: QueryStockTradesReq):
        """查询股票成交"""
        def _respond():
            time.sleep(0.1)
            if 'query_stock_trades' in self.callbacks:
                self.callbacks['query_stock_trades'](seq, [])
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    def queryStockPositionsWithSeq(self, seq: int, req: QueryStockPositionsReq):
        """查询股票持仓"""
        def _respond():
            time.sleep(0.1)
            if 'query_stock_positions' in self.callbacks:
                self.callbacks['query_stock_positions'](seq, [])
        
        thread = threading.Thread(target=_respond)
        thread.daemon = True
        thread.start()
        
    # 其他查询方法的模拟实现
    def queryStockOrderWithSeq(self, seq: int, req):
        """查询单个股票订单"""
        self.queryStockOrdersWithSeq(seq, req)
        
    def queryStockPositionWithSeq(self, seq: int, req):
        """查询单个股票持仓"""
        self.queryStockPositionsWithSeq(seq, req) 