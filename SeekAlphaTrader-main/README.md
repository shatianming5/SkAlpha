# **SeekAlphaTrader**

## SeekAlpha量化交易回测、模拟盘、以及实盘使用说明

### 1. 运行本项目需先开通券商账号以及量化交易（QMT）权限。
- 以国金证券为例，首先下载国金佣金宝APP，开通券商账户。
- 向券商申请量化交易权限（需要10万元保证金，开通后可取出），开通成功后会获得QMT的用户名与密码。
- 下载券商的QMT客户端: 
    - [国金证券QMT交易客户端](https://download.gjzq.com.cn/gjty/organ/gjzqqmt.rar)
    - [国金证券QMT交易测试客户端](https://download.gjzq.com.cn/temp/organ/gjzqqmt_ceshi.rar)


### 2. 下载XtQuant（当前仓库已内置，可跳过）
XtQuant是基于迅投MiniQMT衍生出来的一套完善的Python策略运行框架，以Python库的形式提供策略交易所需要的**行情**和**下单API接口**。需要在[XtQuant下载页面](https://dict.thinktrader.net/nativeApi/download_xtquant.html?id=qbI1Ki)下载并解压到本项目根目录下。


### 3. 安装项目依赖
- 安装Python环境
```
conda create -n qmt python=3.11
conda activate qmt
pip install -r requirements.txt
```

- 新建一个`.env`文件，配置好QMT路径等环境变量到`.env`文件中，例如：
```
QMT_PATH="D:\\国金证券QMT交易端\\userdata_mini"
QMT_EXE_PATH="D:\\国金证券QMT交易端\\bin.x64\\XtMiniQmt.exe"
QMT_ACCOUNT_ID="8883XXX488"

SIMULATE_QMT_PATH="D:\\国金QMT交易端模拟\\userdata_mini"
SIMULATE_QMT_EXE_PATH="D:\\国金QMT交易端模拟\\bin.x64\\XtMiniQmt.exe"
SIMULATE_QMT_ACCOUNT_ID="3999XXXX"
```


### 4. 回测系统使用


多因子策略回测（可选baostock数据或QMT数据，数据自动下载）
```
python backtest_alphatable_xgboost.py
```
使用QMT随后程序会自动启动QMT客户端登录界面，登录即可，随后再运行一次



### **TO-DOs**
- 加入因子暴露、风险控制等评估
- 考虑买入、卖出未完全成交的情况，策略兼容回测与实盘
- 实现实盘交易代码