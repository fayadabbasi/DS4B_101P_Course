

#### Basic Financial Plotting ----

# import 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg", force=True)
%matplotlib inline
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import bt
import talib

#### Financial Trading with the bt package ----


# 1. get historical data
bt_data = bt.get('goog:Low, goog:High, goog:Close',
                    start='2019-1-1', 
                    end='2021-1-1',
                    ticker_field_sep=':')
bt_data.columns = ['Low','High','Close']
bt_data.head()
# bt_data.reset_index()

# 2. define trading strategy
bt_strategy = bt.Strategy('Trade_Weekly',
				[bt.algos.RunWeekly(), # run the trading algo weekly
				bt.algos.SelectAll(), # use all the data
				bt.algos.WeighEqually(), # maintain equal weighting
				bt.algos.Rebalance() # rebalance the portfolio
                                ] 
                            )

# 3. backtest the strategy w data
bt_test = bt.Backtest(bt_strategy, bt_data)
bt_res = bt.run(bt_test)

# 4. evaluate the result
bt_res.plot(title="Backtest result")
bt_res.get_transactions() # get transaction results

#### Trend Indicators moving average ----

# Three types of indicators:
	# 1. Trend indicator - direction and strength of trend (MA)
	# 2. Momentum indicator - velocity of price movement (rel strength index)
	# 3. Volatility indicator - magn of price deviation (bollinger bands)

bt_data['SMA_short_goog'] = talib.SMA(bt_data['Close'], timeperiod=10)
bt_data['SMA_long_goog'] = talib.SMA(bt_data['Close'], timeperiod=50)


plt.plot(bt_data['SMA_short_goog'], label='SMA_short_goog')
plt.plot(bt_data['SMA_long_goog'], label='SMA_long_goog')
plt.plot(bt_data['Close'], label='close_goog')
plt.legend()
plt.title('SMAs')
plt.show()


# EMA has higher weight to most recent price
bt_data['EMA_short_goog'] = talib.EMA(bt_data['Close'], timeperiod=10)
bt_data['EMA_long_goog'] = talib.EMA(bt_data['Close'], timeperiod=50)
plt.plot(bt_data['EMA_short_goog'], label='EMA_short_goog')
plt.plot(bt_data['EMA_long_goog'], label='EMA_long_goog')
plt.plot(bt_data['Close'], label='close_goog')
plt.legend()
plt.title('EMAs')
plt.show()


#### Strength indicators: ADX ----

# ADX - average directional movement index; no trend under 25, trend over 25, strong trend over 50
	# +DI: uptrend, -DI: downtrend

bt_data['ADX'] = talib.ADX(bt_data['High'], bt_data['Low'], bt_data['Close'], timeperiod=14)

bt_data.tail()

fig, (ax1, ax2) = plt.subplots(2)
ax1.set_ylabel('Price')
ax1.plot(bt_data['Close'])
ax2.set_xlabel('ADX')
ax2.plot(bt_data['ADX'])

ax1.set_title('Price and ADX')
plt.show()

#### Momentun Indicators: RSI ----

# RSI measures from 0 to 100; > 70 is overbought and < 30 is oversold
bt_data['RSI'] = talib.RSI(bt_data['Close'], timeperiod=14)
fig, (ax1, ax2) = plt.subplots(2)
ax1.set_ylabel('Price')
ax1.plot(bt_data['Close'])
ax2.set_ylabel('RSI')
ax2.plot(bt_data['RSI'])
ax1.set_title('Price and RSI')
plt.show()


#### Volatility Indicator: Bollinger Bands ----

# RSI measures from 0 to 100; > 70 is overbought and < 30 is oversold
# price volatility by three lines, simple moving average and standard deviation above and below
	# the wider the bollinger bands, the more volatile the asset
upper, mid, lower = talib.BBANDS(bt_data['Close'], nbdevup=2, nbdevdn=2, timeperiod=20)
plt.plot(bt_data['Close'], label='Price')
plt.plot(upper, label='Upper band', alpha=0.2)
plt.plot(mid, label='Middle band', alpha=0.2)
plt.plot(lower, label='Lower band', alpha=0.2)
plt.title('Bollinger Bands')
plt.legend()
plt.show()


#### Trading Signals ----

# trading signals: trigger to go long our short based on:
	# one or many technical indicators
	# combination of market data and indicators
	# ex price > SMA -> go long asset; price < SMA -> sell or go short asset

price_data = bt_data[['Close']]
price_data

sma = price_data.rolling(20).mean()
# OR
sma = talib.SMA(price_data, timeperiod=20)

bt_strategy=bt.Strategy('AboveEMA',
				[bt.algos.SelectWhere(price_data>sma),
				 bt.algos.WeighEqually(),
				 bt.algos.Rebalance()])
# Create the backtest and run it
bt_backtest = bt.Backtest(bt_strategy, price_data)
bt_result = bt.run(bt_backtest)
# Plot the backtest result
bt_result.plot(title='Backtest result')
plt.show()


#### Trend Following Strategies ----

# trend following: two EMA cross over
EMA_short = talib.EMA(price_data['Close'], timeperiod=10).to_frame()
EMA_long = talib.EMA(price_data['Close'], timeperiod=40).to_frame()
signal= EMA_long.copy()
signal[EMA_long.isnull()]=0
signal[EMA_short>EMA_long]=1
signal[EMA_short<EMA_long]=-1
combined_df=bt.merge(signal, price_data, EMA_short, EMA_long)
combined_df.columns=['Signal','Price','EMA_Short','EMA_long']
combined_df.plot(secondary_y=['Signal'])

bt_strategy = bt.Strategy('EMA_Crossover',[bt.algos.WeighTarget(signal), bt.algos.Rebalance()])
bt_backtest=bt.Backtest(bt_strategy, price_data)
bt_result=bt.run(bt_backtest)
bt_result.plot(title='Backtest result')

#### Mean Reversion Strategy ----

# RSI > 70 suggests asset is overbought -- sell signal
# RSI < 30 suggests asset is oversold -- buy signal

stock_rsi = talib.RSI(price_data['Close']).to_frame()
signal = stock_rsi.copy()
signal[stock_rsi.isnull()] = 0
# construct signal
signal[stock_rsi < 30] = 1
signal[stock_rsi > 70] = -1
signal[(stock_rsi<=70) & (stock_rsi>=30)] = 0
combined_df = bt.merge(signal, bt_data['Close'])
combined_df.columns = ['Signal','Price']
# plot signal w price
combined_df.plot(secondary_y = ['Signal'])
# plot RSI
stock_rsi.plot()
plt.title('RSI')
# define strategy
bt_strategy = bt.Strategy('RSI_MeanReversion', [bt.algos.WeighTarget(signal), 
						bt.algos.Rebalance()])
bt_backtest = bt.Backtest(bt_strategy, price_data)
bt_result = bt.run(bt_backtest)
bt_result.plot(title='Backtest result')

#### Strategy Optimization and Benchmarking ----

def signal_strategy(ticker, period, name, start='2018-4-1', end='2020-11-1'):
	price_data = bt.get(ticker, start=start, end=end)
	sma = price_data.rolling(period).mean()
	bt_strategy = bt.Strategy(name, [bt.algos.SelectWhere(price_data>sma), 
					 bt.algos.WeighEqually(), 
					 bt.algos.Rebalance()])
	return bt.Backtest(bt_strategy, price_data)

# Strategy optimization
ticker = 'aapl'
sma20 = signal_strategy(ticker, period=20, name='SMA20')
sma50 = signal_strategy(ticker, period=50, name='SMA50')
sma100 = signal_strategy(ticker, period=100, name='SMA100')
bt_results = bt.run(sma20, sma50, sma100)
bt_results.plot(title='Strategy optimization')

# define benchmark
def buy_and_hold(ticker, name, start='2018-11-1', end='2020-12-1'):
	price_data = bt.get(ticker, start=start, end=end)
	bt_strategy = bt.Strategy(name, [bt.algos.RunOnce(), 
					 bt.algos.SelectAll(), 
					 bt.algos.WeighEqually(), 
					 bt.algos.Rebalance()])
	return bt.Backtest(bt_strategy, price_data)

benchmark = buy_and_hold(ticker, name='benchmark')
bt_results = bt.run(sma20, sma50, sma100, benchmark)
bt_results.plot(title='Strategy Benchmarking')

#### Strategy Return Analysis ----

# get all backtest stats
resInfo = bt_result.stats
print(resInfo.index)

# strategy performance evaluation - rate of return
print('Daily return: %.4f'% resInfo.loc['daily_mean'])
print('Monthly return: %.4f'% resInfo.loc['monthly_mean'])
print('Yearly return: %.4f'% resInfo.loc['yearly_mean'])

# strategy performance evaluation - compound annual growth rate
print('CAGR: %.4f'% resInfo.loc['cagr'])

bt_result.plot_histograms(bins=50, freq='w')
lookback_returns = bt_result.display_lookback_returns()
print(lookback_returns)


#### Drawdown ----

# peak to trough decline is drawdown; downside volatility

max_drawdown = resInfo.loc['max_drawdown']
print('Maximium drawdown: %.2f'% max_drawdown)
avg_drawdown = resInfo.loc['avg_drawdown']
print('Average drawdown: %.2f'% avg_drawdown)
avg_drawdown_days = resInfo.loc['avg_drawdown_days']
print('Average drawdown days: %.0f'% avg_drawdown_days)

# CALMAR ratio: California Managed Accounts Report: CAGR/Max drawdown (over 3 is considered excellent)
resInfo = bt_result.stats
cagr = resInfo.loc['cagr']
max_drawdown = resInfo.loc['max_drawdown']
calmar_calc = cagr/max_drawdown * (-1)
print('Calmar Ratio Calculated: %.2f'% calmar_calc)
# can also get this directly from bt_results.stats
calmar = resInfo.loc['calmar']
print('Calmar Ratio: %.2f'% calmar)


#### Sharpe Ratio and Sortino Ratio

resInfo = bt_result.stats
print('Sharpe ratio daily: %.2f'% resInfo.loc['daily_sharpe'])
print('Sharpe ratio monthly: %.2f'% resInfo.loc['monthly_sharpe'])
print('Sharpe ratio annually: %.2f'% resInfo.loc['yearly_sharpe'])
# OR
annual_return = resInfo.loc['yearly_mean']
volatility = resInfo.loc['yearly_vol']
sharpe_ratio = annual_return / volatility
print('Sharpe ratio annually: %.2f'% sharpe_ratio)
# Sharpe ratio penalizes both good and bad volatility, upside volatility can skew ratio downward
# Sortino ratio uses only downside volatility in denominator to address issue w Sharpe ratio
resInfo = bt_result.stats
print('Sortino ratio daily: %.2f'% resInfo.loc['daily_sortino'])
print('Sortino ratio monthly: %.2f'% resInfo.loc['monthly_sortino'])
print('Sortino ratio annually: %.2f'% resInfo.loc['yearly_sortino'])
