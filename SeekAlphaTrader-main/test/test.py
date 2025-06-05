"""
测试命令: python -m pytest app/core/test/test.py
"""
import pytest
import numpy as np
import pandas as pd
from app.core.portfolio_manager.function import *
from time import time
from app.core.portfolio_manager.alphaparser import AlphaParser



@pytest.mark.parametrize("expression,expected_type", [
    ("SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT, 20, CLOSE<DELAY(CLOSE,1))", pd.Series),
    ("RANK(CAP) < 0.3 && SMA(CLOSE, 10) < SMA(CLOSE, 30) ? ((SMA(CLOSE, 30) ? -1 : 1)) : SMA(CLOSE, 10) > SMA(CLOSE, 30) ? (1.5) : (SMA(CLOSE, 10) < SMA(CLOSE, 30) ? -1.5 : 0)", pd.Series),
    ("(0.6 * RANK(CAP)) + (0.2*TSRANK(DELTA(CLOSE, 20) / DELAY(CLOSE, 20), 20)) + (0.2 * TSRANK(DELTA(VOLUME, 20) / DELAY(VOLUME, 20), 20))", pd.Series),
    ("CORR(RANK(DELTA(LOG(VOLUME), 5)), RANK(DELTA(LOG(CLOSE), 5)), 5)", pd.Series)
    # 在这里添加更多的表达式和期望的返回类型
])


# 测试解析表达式的功能
def test_parse_expression_basic(expression, expected_type):
    parser = AlphaParser()
    parsed_expr = parser.parse_expression(expression)
    result = parser.check_expr(parsed_expr)
    assert isinstance(result, pd.DataFrame) or isinstance(result, pd.Series), "结果应该是一个DataFrame或Series"





    
    

