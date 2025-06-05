'''
Main file for backtesting a strategy on historical stock data.
'''
import datetime
import pickle
import os
import re
import argparse
import json
import pyparsing
import requests
import json
import shutil
from typing import Text, List, Dict, Tuple

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer

from data_manager.dataloader import BaoStockLoader, QMTDataLoader
from portfolio_manager.portfolio_management import AlphaGPTPortfolioManager
from evaluator.performance_evaluation import PerformanceEvaluator
from test.visualization import draw_figures
from portfolio_manager.action_management import ActionManager
from output_manager import OutputManager
from expr_parser import parse_expression
from function_lib import *

def calculate_ic(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

def backtest(exprs:Dict[str, str]=None, date_split:Dict[str, str]=None, **kwargs) -> dict:
    '''
    回测函数，输入参数为策略参数，输出为回测结果
    '''
    # 初始化输出管理器
    output_manager = OutputManager()
    
    try:
        train_start_time = date_split['train_start_time']
        train_end_time = date_split['train_end_time']
        val_start_time = date_split['val_start_time']
        val_end_time = date_split['val_end_time']
        test_start_time = date_split['test_start_time']
        test_end_time = date_split['test_end_time']

    # 加载策略参数，根据用户选择的股票池获取对应股票指数的数据集ID
    action_manager = ActionManager(**kwargs)

    try:
        assert test_end_time <= datetime.datetime.now().strftime('%Y-%m-%d'), '测试结束时间不能大于当前时间'
        backtest_start_time = pd.to_datetime(train_start_time)
        backtest_end_time = pd.to_datetime(test_end_time)

        train_start_time = pd.to_datetime(train_start_time)
        train_end_time = pd.to_datetime(train_end_time)
        val_start_time = pd.to_datetime(val_start_time)
        val_end_time = pd.to_datetime(val_end_time)
        test_start_time = pd.to_datetime(test_start_time)
        test_end_time = pd.to_datetime(test_end_time)
        index_code = 'sh.000905'
        

        # loader = BaoStockLoader(
        #     data_dir='./data_baostock', 
        # )

        loader = QMTDataLoader(
            data_dir='./data_qmt',
            simulate_client=True,
        )

        # 获取指数成分股
        print(f"正在加载{index_code}成分股列表...")
        constituent_stock_codes = loader.load_index_stocklist_timerange(index_code, backtest_start_time, backtest_end_time)
        print(f"成分股列表加载完成，共{len(constituent_stock_codes)}个时间点")


        # 获取指数数据
        benchmark_data = loader.load_stock_data(code=index_code, start_date=backtest_start_time, end_date=backtest_end_time)
        benchmark_data.index = pd.to_datetime(benchmark_data.index)

        

        ########################
        ###  获取个股行情数据  ###
        ########################
        print("正在加载个股行情数据...")
        combined_df = loader.load_stock_price_timerange(constituent_stock_codes, backtest_start_time, backtest_end_time, use_cache=True)
        print(f"行情数据加载完成，共{len(combined_df.index.get_level_values('instrument').unique())}只股票")



        ########################
        ###  成分股掩码计算  ###
        ########################
        # 1. 使用combined_df的索引创建mask
        constituent_mask = pd.DataFrame(False, index=combined_df.index, columns=['is_constituent'])
        
        # 2. 根据constituent_stock_codes填充mask
        update_dates = pd.Series(constituent_stock_codes.keys())
        update_dates.sort_values(inplace=True)
        for timestamp, stocks in constituent_stock_codes.items():
            # 找到下一个成分股更新日期
            if timestamp == update_dates.iloc[-1]:
                next_update_timestamp = backtest_end_time
            else:
                next_update_timestamp = update_dates[update_dates > timestamp].iloc[0]

            idx = pd.IndexSlice[timestamp:next_update_timestamp, stocks]
            constituent_mask.loc[idx, 'is_constituent'] = True


        #####################
        ###  因子计算部分  ###
        #####################
        print("正在计算因子...")
        # 计算其他字段
        combined_df.loc[:, 'return'] = combined_df.loc[:, 'close'].groupby('instrument').shift(0) / combined_df.loc[:, 'close'].groupby('instrument').shift(1) - 1
        
        # 计算因子
        for i, (name, expr) in enumerate(exprs.items()):
            expr = parse_expression(expr)
            for col in combined_df.columns:
                expr = expr.replace('$'+col, f"combined_df[\'{col}\']")

            # expr = f"ZSCORE({expr})"
            combined_df[name] = eval(expr).astype(float)
            # result = combined_df[name]

        # 计算基础feature
        base_feature = {
            "intraday_return": "($close-$open)/$open-1", 
            "return": "$close/DELAY($close, 1)-1", 
            "relative_volume": "$volume/TS_MEAN($volume, 20)-1", 
            "amplitude": "($high-$low)/DELAY($close, 1)",
        }
        for feature, expr in base_feature.items():
            expr = parse_expression(expr)
            for col in combined_df.columns:
                expr = expr.replace('$'+col, f"combined_df[\'{col}\']")
            combined_df[feature] = eval(expr)

        print(f"因子计算完成，共计算{len(exprs)}个因子")
        
        #######################
        ###  ML模型训练推理部分  ###
        #######################
        print("正在训练机器学习模型...")

        # 构建训练feature
        feature_cols = list(exprs.keys()) + list(base_feature.keys())
        feature_df = combined_df[feature_cols]
        feature_df = feature_df.fillna(0)

        # 构建label: 5天后的收益
        feature_df.loc[:, 'label'] = combined_df.loc[:, 'close'].groupby('instrument').shift(-4) / combined_df.loc[:, 'close'].groupby('instrument').shift(0) - 1
        
        # 对特征进行横截面标准化
        def cross_sectional_zscore(df, cols):
            # 按日期分组，对每个时间点进行标准化
            return df.groupby(level=0)[cols].transform(lambda x: (x - x.mean()) / x.std())
        
        # 对特征进行横截面标准化
        feature_df.loc[:, feature_cols] = cross_sectional_zscore(feature_df, feature_cols)
        feature_df.loc[:, 'label'] = cross_sectional_zscore(feature_df, ['label'])

        # 划分训练集和测试集
        train_df = feature_df.loc[train_start_time:train_end_time]
        val_df = feature_df.loc[val_start_time:val_end_time]
        test_df = feature_df.loc[test_start_time:test_end_time]

        # 准备训练数据
        feature_cols = [col for col in train_df.columns if col != 'label']
        
        # 不再需要全局标准化，因为已经进行了横截面标准化
        train_x = train_df[feature_cols]
        train_y = train_df['label']
        
        val_x = val_df[feature_cols]
        val_y = val_df['label']
        
        test_x = test_df[feature_cols]
        test_y = test_df['label']

        # 使用fillna填充缺失值，保持维度一致
        train_x = train_x.fillna(0)  # 使用0填充特征缺失值
        train_y = train_y.fillna(0)  # 使用0填充标签缺失值
        
        val_x = val_x.fillna(0)
        val_y = val_y.fillna(0)
        
        test_x = test_x.fillna(0)
        test_y = test_y.fillna(0)

        # 设置模型参数
        params = {
            'objective': 'reg:squarederror',  # 目标函数：使用均方误差(MSE)作为回归问题的损失函数，适合预测连续值（如股票收益率）
            'eval_metric': 'rmse',            # 评估指标：使用均方根误差(RMSE)来评估模型性能，用于监控训练过程
            'booster': 'gbtree',              # 基学习器类型：使用决策树作为基学习器，适合处理非线性关系
            'max_depth': 5,                   # 决策树最大深度：控制树的复杂度，值越大模型越复杂，但可能过拟合
            'learning_rate': 0.1,             # 学习率：控制每棵树的权重缩减，值越小模型学习越慢但更稳定
            'subsample': 0.8,                 # 样本采样比例：训练每棵树时随机使用80%的训练数据，防止过拟合
            'colsample_bytree': 0.5,          # 特征采样比例：训练每棵树时随机使用50%的特征，防止模型过度依赖某些特征
            'min_child_weight': 1,            # 叶子节点最小样本权重：控制叶子节点的生成条件，值越大模型越保守
            'verbosity': 0                    # 输出信息详细程度：0表示不输出训练过程信息，可设为1或2查看详细信息
        }
        # 创建数据集
        
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dval = xgb.DMatrix(val_x, label=val_y)

        # 训练模型
        evallist = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evallist,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # 模型推理
        train_pred = model.predict(xgb.DMatrix(train_x))
        val_pred = model.predict(xgb.DMatrix(val_x))
        test_pred = model.predict(xgb.DMatrix(test_x))

        # 计算各个集合的MSE
        train_mse = mean_squared_error(train_y, train_pred)
        val_mse = mean_squared_error(val_y, val_pred)
        test_mse = mean_squared_error(test_y, test_pred)

        # 特征重要性
        importance_dict = model.get_score(importance_type='weight')
        # 确保所有特征都有对应的重要性值，没有的设为0
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': [importance_dict.get(f, 0) for f in feature_cols]
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # 计算IC
        train_ic = calculate_ic(train_y, train_pred)
        val_ic = calculate_ic(val_y, val_pred)
        test_ic = calculate_ic(test_y, test_pred)

        # 统一输出模型训练结果（只在最终结果中输出一次）
        model_results = {
            'train_mse': train_mse,
            'val_mse': val_mse, 
            'test_mse': test_mse,
            'train_ic': train_ic,
            'val_ic': val_ic,
            'test_ic': test_ic,
            'top_features': feature_importance.head(10)
        }

        # 将预测结果存储为新的DataFrame
        pred_df = pd.DataFrame(index=feature_df.index, columns=['pred'])
        pred_df.loc[train_df.index, 'pred'] = train_pred
        pred_df.loc[val_df.index, 'pred'] = val_pred
        pred_df.loc[test_df.index, 'pred'] = test_pred

        # 生成的预测值用于回测（只使用测试集时间段的预测结果）
        alpha_table = pred_df.loc[test_start_time:test_end_time, 'pred'].unstack()
        all_instruments = pred_df.index.get_level_values('instrument').unique()
        alpha_table = alpha_table.reindex(columns=all_instruments)
        # 对齐index
        alpha_table.index = benchmark_data.loc[test_start_time:test_end_time].index

        print("模型训练完成，正在准备回测数据...")
        
        ####################
        ##### 回测部分  #####
        ####################
        # 获取测试区间的未复权数据用于交易，默认是开盘价
        deal_price_data = pd.DataFrame({code: combined_df.xs(code, level=1)['close_unadj'] for code in all_instruments}).set_index(benchmark_data.index).loc[test_start_time:test_end_time, alpha_table.columns]
        # 获取测试区间的后复权价格数据，用于计算收益
        postadj_close = pd.DataFrame({code: combined_df.xs(code, level=1)['close'] for code in all_instruments}).set_index(benchmark_data.index).loc[test_start_time:test_end_time, alpha_table.columns]
        # 获取测试区间的基准指数数据
        benchmark_data = benchmark_data.loc[test_start_time:test_end_time]
        # 对价格实施掩码，以在回测时，不交易非成分股
        deal_price_data = deal_price_data.mask(~constituent_mask['is_constituent'].unstack().loc[test_start_time:test_end_time])
        postadj_close = postadj_close.mask(~constituent_mask['is_constituent'].unstack().loc[test_start_time:test_end_time])
        # 初始化回测组件
        action_manager = ActionManager(**kwargs)
        portfolio_manager = AlphaGPTPortfolioManager(update_freq=kwargs.get('update_freq', 'M'), max_pos_each_stock=kwargs.get('max_pos_each_stock', 0.1))
        performance_evaluator = PerformanceEvaluator()
        # 核心回测函数
        print("正在执行回测...")
        results = performance_evaluator.backtest_factor_table(alpha_table, deal_price_data, postadj_close, portfolio_manager, action_manager)
        print("回测执行完成")

        results.update({
            'expr': kwargs.get('expr', ''),
            'BENCHMARKINDEX': benchmark_data['close'],
            'PRICE': postadj_close,
            'start_cash': action_manager.start_cash,
        })

        # 计算评估指标
        print("正在计算评估指标...")
        results_to_save = performance_evaluator.calculate_evaluation_metrics(results)
        # action参数也保存，以供复现
        results_to_save.update({k: v for k, v in action_manager.__dict__.items() if not isinstance(v, pd.Series)})
        
        # 使用统一输出管理器保存因子明细数据包
        print("正在保存因子明细数据...")
        factor_zip_path = output_manager.create_factor_package(
            trade_signals_df=results['trade_signals'],
            portfolios_df=results['total_portfolios'],
            alpha_table_df=alpha_table
        )
        
        # 保存回测报告
        print("正在保存回测报告...")
        report_path = output_manager.save_backtest_report(results_to_save)
        
        # 更新结果中的文件路径
        results_to_save.update({
            "factor_zip_path": factor_zip_path,
            "report_path": report_path,
            "output_manager": output_manager
        })
        
        return results_to_save
        
    except Exception as e:
        print(f"回测执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回默认的错误结果
        return {
            'test_ic': 0.0,
            'test_rank_ic': 0.0,
            'info_ratio': 0.0,
            'averaged_annualized_roi': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'monthly_win_rate': 0.0,
            'overall_trading_win_rate': 0.0,
            'turnover': 0.0,
            'algorithm_volatility': 0.0,
            'error': str(e)
        }
        
    finally:
        # 清理临时文件
        if 'output_manager' in locals():
            output_manager.clean_temp_files()
        

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Trading strategy backtester")
    # parser.add_argument('--expr', 
    #                     default="TS_STD($return, 20)",
    #                     type=str)
    # args = parser.parse_args()
    
    results_to_save = backtest(
        exprs={
            "Combined-Price-Volume-Dynamics-Factor-V2":"RANK(TS_CORR($close, $volume, 20)) * RANK(TS_SUM($close, 30))",
            "Smart_Volume_Cluster_Composite": "(TS_STD($close,5)/(TS_STD($close,20)+1e-8)) * ($volume > TS_QUANTILE($volume,20,0.9))",
            "RangeVolume_SlopeProduct_5D": "COUNT((($high - $low) < TS_MEAN(($high - $low),5)) && ($volume < TS_MEAN($volume,5)),5)",
            "Dynamic_Volatility_Bands_Momentum_Stochastic_Oscillator_Factor": "(($close - TS_MIN($low, 14)) / (TS_MAX($high, 14) - TS_MIN($low, 14) + 1e-8)) * 100",
            "Adjusted_Normalization_Factor_10D": "RANK(TS_ZSCORE(TS_VAR(DELAY($close, 5), 10))) * TS_ZSCORE(DELTA($close, 7))",
            "Drawdown-Minimization-Factor": "1 - ($high / TS_MAX($high, 20) + 1e-8)",
            "Price_Volume_Trend_Factor_V2": "(EMA($close, 5) - EMA($close, 20)) * LOG(EMA($volume, 5))",
            "BollBand_Width_Factor": "(BB_UPPER($close, 20) - BB_LOWER($close, 20)) / (EMA($high - $low, 15) + 1e-8)",
            "Trend_Following_Mean_Reversion_Factor_10D": "TS_RANK(SMA($close - $open, 5, 1), 10) - TS_RANK(SMA(DELTA($close, 1), 7, 1), 10)",
            "Volume_Price_Volatility_Factor": "ABS(TS_CORR($volume, $return, 10)) * SQRT(TS_STD($close - $open, 20) + 1e-8)",
            "Fine_Tuned_Mean_Reversion_Factor_7D": "TS_RANK(ABS(DELTA($close, 1)), 5) + TS_RANK(ZSCORE(DELAY($close, 3)), 5)",
            "Moving_Average_Window_Optimization_Factor": "($close > DELAY($close, 1)) ? (EMA($close, 2) + EMA($close, 5)) : (EMA($close, 10) - EMA($close, 15))",
            "Adjusted_Normalization_Factor_10D": "RANK(SQRT(TS_VAR(DELAY($close, 5), 10))) * ZSCORE(DELTA($close, 7))",
            },
        date_split={
            'train_start_time': '2015-01-01',
            'train_end_time': '2023-12-31',
            'val_start_time': '2024-01-01',
            'val_end_time': '2024-04-30',
            'test_start_time': '2024-05-01',
            'test_end_time': '2025-05-29'
            },
        stop_loss_rate=0.5,
        stop_profit_rate=0.4,
        start_cash=1e7,
        position_size=1.0,
        update_freq=5,
        max_pos_each_stock=0.05,
        stock_pool='中证500'
        )

    # 获取输出管理器
    output_manager = results_to_save.get('output_manager')
    
    print("正在生成图表...")
    # 使用统一的输出路径
    figure_path = output_manager.get_figure_path()
    draw_figures(results_to_save, figure_path)
    
    # 统一输出最终回测结果摘要
    print("=" * 60)
    print("回测结果摘要")
    print("=" * 60)
    print(f"年化收益率: {results_to_save.get('averaged_annualized_roi', 0)*100:.2f}%")
    print(f"信息比率: {results_to_save.get('info_ratio', 0):.4f}")
    print(f"最大回撤: {results_to_save.get('max_drawdown', 0)*100:.2f}%")
    print(f"总胜率: {results_to_save.get('overall_trading_win_rate', 0)*100:.2f}%")
    print(f"换手率: {results_to_save.get('turnover', 0)*100:.2f}%")
    print("=" * 60)
    
    # 打印输出文件摘要
    output_manager.print_output_summary()
    
    # 清理旧文件（保留最近7天）
    output_manager.clean_old_outputs(days_to_keep=7)
    
