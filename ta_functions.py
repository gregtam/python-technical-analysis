from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from textwrap import dedent

from IPython.core.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data
import seaborn as sns

blue, green, red, purple, yellow, teal = sns.color_palette('deep')
black = (0, 0, 0)
white = (1, 1, 1)
blues = sns.color_palette('Blues', n_colors=6)[::-1]

def _listify_security(security):
    """If input is a string, convert it to a list. If the input is a
    list, keep it the same.
    """

    if isinstance(security, str):
        return [security]
    elif isinstance(security, list) or isinstance(security, pd.DataFrame):
        return security

def _get_security_names(security_data):
    """Extracts the security names from the merged DataFrame."""

    _col_values = [word.split('_')[-1] for word in security_data.columns]
    return set(_col_values)

def generate_bollinger_columns(sec_df, securities, col_name,
                               bollinger_std, bollinger_len):
    """Creates columns for Bollinger bands and buy signals.

    Inputs:
    df - The merged DataFrame of security data
    securities - The corresponding list of securities, ETFs, etc.
    col_name - Close, Open, etc.
    bollinger_std - The standard deviation of the Bollinger bands
    bollinger_len - The number of days to use for the moving average

    Returns a Pandas DataFrame with the new Bollinger columns
    """
    sec_df = sec_df.copy()

    securities = _listify_security(securities)

    for security in securities:
        input_col = col_name + '_' + security
        # Set column names for bollinger bands
        b_high = col_name + '_' + security + '_bollinger_high'
        b_low = col_name + '_' + security + '_bollinger_low'

        rolling_window = sec_df[input_col].rolling(bollinger_len)
        rolling_mean = rolling_window.mean()
        rolling_std = rolling_window.std()

        sec_df[b_high] = rolling_mean + bollinger_std * rolling_std
        sec_df[b_low] = rolling_mean - bollinger_std * rolling_std

    return sec_df.dropna()

def generate_ma_columns(sec_df, securities, col_name, ndays):
    """Create columns for moving averages and determines when there are
    crossovers.

    Inputs:
    df - The merged DataFrame of security data
    securities - The corresponding list of securities, ETFs, etc.
    col_name - Close, Open, etc.
    ndays - A list of the moving average lengths we want to generate

    Returns a Pandas DataFrame with the new MA columns
    """
    sec_df = sec_df.copy()
    if len(ndays) != 2:
        raise Exception('Length of ndays must be 2.')

    securities = _listify_security(securities)
    
    for security in securities:
        # Add moving average
        input_col = col_name + '_' + security
        for n in ndays:
            final_col = '{}_{}_{}d_ma'.format(col_name, security, n)
            sec_df[final_col] = sec_df[input_col].rolling(n).mean()

        short_ma = sec_df['{}_{}_{}d_ma'.format(col_name, security, ndays[0])]
        long_ma = sec_df['{}_{}_{}d_ma'.format(col_name, security, ndays[1])]

        ma_diff_col_name = 'ma_diff_' + security
        sec_df[ma_diff_col_name] = short_ma - long_ma
        crossover = np.array(sec_df[ma_diff_col_name])[:-1]\
                    * np.array(sec_df[ma_diff_col_name][1:])
        crossover = [0] + crossover.tolist()

        crossover_col_name = 'crossover_' + security
        sec_df[crossover_col_name] = np.sign(crossover)
        # Equal to 1 if crossed over from previous day to current day,
        # i.e., the signs of ma_diff switches
        sec_df[crossover_col_name] = sec_df[crossover_col_name].map({1:0,
                                                                     0:0,
                                                                     -1:1
                                                                    })
        # Make a new variable signal_SECURITY which determines, based
        # off the moving average, whether to buy, sell or do nothing. We
        # buy when short_ma is larger than long_ma
        signal_col_name = 'crossover_signal_' + security
        sec_df[signal_col_name] = (short_ma > long_ma).map({True: 'Buy',
                                                            False: 'Sell'
                                                           })
        # Next, we only do this when a crossover has occurred, so we
        # need to remove pre-existing signals
        sec_df.loc[sec_df[crossover_col_name] == 0, signal_col_name] = 'N/A'
    
    return sec_df.dropna()

def generate_returns(security_data, col_name):
    """Generates the returns of a given security.

    Inputs:
    security_data - A Pandas DataFrame with the security information.
    col_name - The column name to generate returns for

    Returns:
    sec_returns - A Pandas Series of the returns of the given column
    """

    # Get the first security name
    security_names = _get_security_names(security_data)
    if len(security_names) > 1:
        raise(Exception('There can only be one security in the DataFrame'))
    else:
        security_name = list(security_names)[0]

    col_name = col_name + '_' + security_name
    df_col = security_data[col_name]
    df_last_col = df_col.shift(1)

    sec_returns = ((df_col - df_last_col)/df_last_col)
    return sec_returns

def get_security_data(securities, start_date, end_date=date.today(),
                      data_source='google'):
    """Gets all securities in securities and merges them into a
    DataFrame.

    Inputs:
    securities - A string indicating the desired security or a list of
                  strings indicating the desired securities
    end_date - A string indicating the end date of our data
               (Default: date.today())
    data_source - The source of the security data (Default: 'google')
    """

    securities = _listify_security(securities)

    df_list = [data.DataReader(security, data_source=data_source,
                               start=start_date, end=end_date)
                   for security in securities]

    # Append security name to columns
    for security, df in zip(securities, df_list):
        df.columns = ['{}_{}'.format(col, security).lower()
                          for col in df.columns]

        # for s in ['open', 'close']:
        #     col_name = '{}_{}'.format(s, security.lower())
        #     col = df[col_name]
        #     last_col = col.shift(1)
        #     close_returns = ((col - last_col)/last_col)

        #     df['{}_return_{}'.format(s, security.lower())] = close_returns

    merged_df = df_list[0].copy()
    for i in range(1, len(df_list)):
        merged_df = merged_df.join(df_list[i])

    return merged_df

def run_simulation(security_info, start_date, end_date=date.today(),
                   start_cash_amt=10000, data_source='google', data_store=None,
                   **kwargs):
    """Run a trading simulation for a list of securities. This is a
    wrapper around run_simulation_df(). Its purpose is to load the data,
    then run run_simulation_df() on it, so we do not need to load the
    data into a DataFrame ahead of time.

    Inputs:
    security_info - A string representing a single security, a list
                    of securities, ETFs, etc., or a DataFrame containing
                    the already merged data.
    start_date - A string indicating the start date of our data
    end_date - A string indicating the end date of our data
               (Default: date.today())
    start_cash_amt - Starting portfolio cash amount (Default: 10000)
    data_source - The source of the security data (Default: 'google')
    data_store - A data_storage object to hold the security data. If not 
                 specified, then download data each time.
                 (Default: None)
    kwargs - Remaining keyword arguments
    """

    security_info = _listify_security(security_info)

    if data_store is None:
        security_data = get_security_data(security_info, start_date, end_date,
                                          data_source=data_source)

    else:
        df_list = []
        for sec in security_info:
            if sec in data_store.get_securities():
                _df = data_store.load_security_data(sec,
                                                    start_date,
                                                    end_date,
                                                   )
            else:
                _df = data_store.get_security_data(sec,
                                                   start_date,
                                                   end_date,
                                                   data_source=data_source
                                                  )
            df_list.append(_df)

        security_data = df_list[0].copy()

        for i in range(1, len(df_list)):
            security_data = security_data.join(df_list[i])

    start = datetime.now()
    port = run_simulation_df(security_data, start_cash_amt, **kwargs)
    print datetime.now() - start
    return port
   
def run_simulation_df(security_data, start_cash_amt=10000,
                      indicators=dict(ma_crossovers=[5, 10]), verbose=True,
                      plot_options=set(['transactions'])):
    """Runs a trading simulation on a DataFrame containing all of the
    security data information. This is designed to run on the output
    DataFrame of the get_security_data function.

    Inputs:
    security_data - A Pandas DataFrame with the relevant stock data
    start_cash_amt - Starting portfolio cash amount (Default: 10000)
    indicators - A dictionary of which indicators to use, where the keys
                 are strings representing the indicators and the values
                 indicate the parameters associated with the indicators
                 Possible keys: 'ma_crossovers', 'rsi', 'bollinger_std',
                 and 'bollinger_len'.
                 (Default: {'ma_crossovers': [5, 10]})
    verbose - A boolean of whether to print each trade (Default: True)
    plot_options - A set of which plotting options.
                   Possible options: ('transactions', 'ma')
    """

    securities = _get_security_names(security_data)
    if 'ma_crossovers' in indicators:
        security_data = generate_ma_columns(security_data,
                                            securities,
                                            'close',
                                            indicators['ma_crossovers']
                                           )
    if 'bollinger_std' in indicators and 'bollinger_len' in indicators:
        security_data = generate_bollinger_columns(security_data,
                                                   securities,
                                                   'close',
                                                   indicators['bollinger_std'],
                                                   indicators['bollinger_len']
                                                  )

    sec_port = security_portfolio(start_cash_amt, verbose=verbose)

    bought_securities = set()

    def _get_ma_crossovers_price(index, row, security, purchase_price):
        """Checks for MA crossovers, executes transaction if satisfies
        criteria, and updates purchase_price."""
        close_col_name = 'close_' + security
        ma_diff_col_name = 'ma_diff_' + security

        # If ma_diff is positive on a crossover, then it is trending
        # upwards since 50d ma is smaller than 15d ma
        if security not in bought_securities\
                and row[ma_diff_col_name] > 0\
                and sec_port.get_total_cash_amt() > purchase_price:
            # Buy securities
            sec_port.buy_max_securities(security, 
                                        row[close_col_name], 
                                        index 
                                       )
            bought_securities.add(security)
            purchase_price = row[close_col_name]
        elif security in bought_securities\
                and row[ma_diff_col_name] < 0\
                and row[close_col_name] > purchase_price:
            # Sell securities
            sec_port.sell_all_securities(security, 
                                         row[close_col_name],
                                         index
                                        )
            bought_securities.remove(security)

        return purchase_price

    def _get_bollinger_price(index, row, security, purchase_price):
        """Checks for bollinger criteria, executes transaction if
        passes, and updates purchase_price."""
        close_col_name = 'close_' + security
        high_col_name = 'close_{}_bollinger_high'.format(security)
        low_col_name = 'close_{}_bollinger_low'.format(security)

        if security not in bought_securities\
                and row[close_col_name] < row[low_col_name]\
                and sec_port.get_total_cash_amt() > purchase_price:
            # Buy securities
            sec_port.buy_max_securities(security,
                                        row[close_col_name],
                                        index
                                       )
            bought_securities.add(security)
            purchase_price = row[close_col_name]
        elif security in bought_securities\
                and row[close_col_name] > row[high_col_name]:
                # and row[close_col_name] > purchase_price:
            # Sell securities
            sec_port.sell_all_securities(security, 
                                         row[close_col_name],
                                         index
                                        )
            bought_securities.remove(security)

        return purchase_price

    def _run_simulation():
        """Runs through the simulation."""

        purchase_price = 0
        # For each day
        for index, row in security_data.iterrows():
            for security in securities:
                crossover_col_name = 'crossover_' + security
                if 'ma_crossovers' in indicators\
                        and row[crossover_col_name] == 1:
                    purchase_price = _get_ma_crossovers_price(index,
                                                              row,
                                                              security,
                                                              purchase_price
                                                             )

                if 'bollinger_len' in indicators\
                       and 'bollinger_std' in indicators:
                    purchase_price = _get_bollinger_price(index,
                                                          row,
                                                          security,
                                                          purchase_price
                                                         )

    def _plot_simulation():
        """Plot the security and relevant simulation information."""
        for security in securities:
            close_col_name = 'close_' + security
            # If anything should be plotted
            if len(plot_options) > 0:
                # Plot security closing prices
                security_data[close_col_name].plot(c=black)

                if 'ma_crossovers' in indicators and 'ma' in plot_options:
                    ndays = indicators['ma_crossovers']
                    # Plot each moving average crossover
                    for i in xrange(len(ndays)):
                        day = ndays[i]
                        ma_col_name = 'close_{}_{}d_ma'.format(security, day)

                        # Plot moving average columns
                        plt.plot(security_data.index,
                                 security_data[ma_col_name],
                                 c=blues[i]
                                )

                if 'bollinger_std' in indicators\
                        and 'bollinger' in plot_options:
                    # Get bollinger high and low column names
                    high_col_name = 'close_{}_bollinger_high'.format(security)
                    low_col_name = 'close_{}_bollinger_low'.format(security)

                    # Plot bollinger bands
                    plt.plot(security_data.index, security_data[high_col_name],
                             c=black, linestyle='--', alpha=0.5)
                    plt.plot(security_data.index, security_data[low_col_name],
                             c=black, linestyle='--', alpha=0.5)

        if 'transactions' in plot_options:
            for row in sec_port.get_all_transactions().itertuples():
                if row.trans_type == 'Buy':
                    plt.axvline(x=row.date, label='Buy', linewidth=2.5,
                                linestyle='--', c=red)
                elif row.trans_type == 'Sell':
                    plt.axvline(x=row.date, label='Sell', linewidth=2.5,
                                linestyle='--', c=green)

    _run_simulation()
    _plot_simulation()
    return sec_port

def get_buy_sell_signals(security, column, start_date, end_date=date.today(),
                         plot=True, data_source='google', indicators={},
                         signals=[], **kwargs):
    """Gets buy and sell signals given indicators and plots them.

    Inputs:
    security - The ticker symbol
    column - Open, Close, etc.
    start_date - A string representing the start date
    end_date - A string representing the end date (Default: today)
    plot - A boolean of whether to plot the security and signals
           (Default: True)
    data_source - The data source to pull data from (Default: 'google')
    indicators - A dictionary of which indicators to plot, where the keys
                 are strings representing the indicators and the values
                 indicate the parameters associated with the indicators
                 Possible keys: 'ma_crossovers', 'rsi', 'bollinger_std',
                 and 'bollinger_len'
                 (Default: {})
    signals - A list of signals to plot. Options: 'buy' and 'sell'.
              (Default: {})

    Returns:
    signal_df - A DataFrame of the buy and sell signals
    """

    sec_df = get_security_data(security, start_date, end_date,
                               data_source=data_source)

    desired_column = column + '_' + security.lower()

    # Plot the security price
    if plot:
        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.plot(sec_df.index, sec_df[desired_column], c=black)
        else:
            plt.plot(sec_df.index, sec_df[desired_column], c=black)

    # Generate the moving averages for the security
    if 'ma_crossovers' in indicators:
        ma_crossovers = indicators['ma_crossovers']
        for nday in ma_crossovers:
            ma_name = '{}_ma'.format(nday)
            sec_df[ma_name] = sec_df[desired_column].rolling(nday).mean()

    if 'bollinger_len' in indicators and 'bollinger_std' in indicators:
        bollinger_len = indicators['bollinger_len']
        bollinger_std = indicators['bollinger_std']

        # Get rolling mean
        rolling_mean = sec_df[desired_column].rolling(bollinger_len).mean()

        # Get rolling standard deviation
        rolling_std = sec_df[desired_column].rolling(bollinger_len).std()

        # Define the upper and lower bollinger bands
        sec_df['bollinger_high'] = rolling_mean + bollinger_std * rolling_std
        sec_df['bollinger_low'] = rolling_mean - bollinger_std * rolling_std

        # Plot the upper and lower bollinger bands
        if plot:
            if 'ax' in kwargs:
                ax.plot(sec_df.index, sec_df.bollinger_high,
                        c=black, linestyle='--', alpha=0.5)
                ax.plot(sec_df.index, sec_df.bollinger_low,
                        c=black, linestyle='--', alpha=0.5)
            else:
                plt.plot(sec_df.index, sec_df.bollinger_high,
                         c=black, linestyle='--', alpha=0.5)
                plt.plot(sec_df.index, sec_df.bollinger_low,
                         c=black, linestyle='--', alpha=0.5)

    # Plot moving averages
    if 'ma_crossovers' in indicators and plot:
        for i in np.arange(len(ma_crossovers)):
            nday = ma_crossovers[i]
            if 'ax' in kwargs:
                ax.plot(sec_df.index, sec_df['{}_ma'.format(nday)],
                        c=blues[i], alpha=0.5)
            else:
                plt.plot(sec_df.index, sec_df['{}_ma'.format(nday)],
                         c=blues[i], alpha=0.5)
    signal_df = pd.DataFrame()
    # Plot buy signals
    if 'buy' in signals:
        if 'ma_crossovers' in indicators:
            pass
        if 'bollinger_len' in indicators and 'bollinger_std' in indicators:
            # buy_df = sec_df.query('{} < bollinger_low'.format(desired_column))
            buy_df = sec_df.query(desired_column + ' < bollinger_low').copy()
            buy_df.insert(0, 'signal', 'buy')
            buy_df.insert(0, 'security', security.upper())
            # buy_df['signal'] = 'buy'
            # buy_df['security'] = security.upper()
            signal_df = pd.concat([signal_df, buy_df])
            if plot:
                for row in buy_df.itertuples():
                    if 'ax' in kwargs:
                        ax.axvline(x=row.Index, label='Buy', linewidth=2.5,
                                   linestyle='--', c=red)
                    else:
                        plt.axvline(x=row.Index, label='Buy', linewidth=2.5,
                                    linestyle='--', c=red)
    # Plot sell signals
    if 'sell' in signals:
        if 'ma_crossovers' in indicators:
            pass
        if 'bollinger_len' in indicators and 'bollinger_std' in indicators:
            # sell_df = sec_df.query('{} > bollinger_high'.format(desired_column))
            sell_df = sec_df.query(desired_column + ' > bollinger_high').copy()
            sell_df.insert(0, 'signal', 'sell')
            sell_df.insert(0, 'security', security.upper())
            # sell_df['signal'] = 'sell'
            # sell_df['security'] = security.upper()
            signal_df = pd.concat([signal_df, sell_df])
            if plot:
                for row in sell_df.itertuples():
                    if 'ax' in kwargs:
                        ax.axvline(x=row.Index, label='Sell', linewidth=2.5,
                                   linestyle='--', c=green)
                    else:
                        plt.axvline(x=row.Index, label='Sell', linewidth=2.5,
                                    linestyle='--', c=green)

    return signal_df

def plot_trades(sec_port):
    """Plots the trades."""

    trans_df = sec_port.get_all_transactions()
    for row in trans_df.itertuples():
        if row.trans_type == 'Buy':
            plt.axvline(row.date, linestyle='--', linewidth=3, color=red)
        elif row.trans_type == 'Sell':
            plt.axvline(row.date, linestyle='--', linewidth=3, color=green)

def plot_rsi(security, column, start_date, end_date=date.today(), ndays=14,
             thresholds=[20, 80], **kwargs):
    """Plots a single security.

    Inputs:
    security - The ticker symbols
    column - open, close, etc.
    start_date - A string representing the start date
    end_date - A string representing the end date (Default: today)
    ndays - The number of days to use as a lookback period (Default: 14)
    """

    def _rsi_agg(security_array):
        """Returns RSI."""
        
        # Get differences of the security
        sec_diff = np.diff(security_array)
        pos_array = sec_diff[sec_diff > 0]    
        neg_array = sec_diff[sec_diff < 0]    
        
        if len(pos_array) > 0:
            pos_mean = pos_array.mean()
        if len(neg_array) > 0:
            neg_mean = -neg_array.mean()
        
        if len(neg_array) == 0 and len(pos_array) == 0:
            rs = 50
        elif len(neg_array) == 0 and len(pos_array) > 0:
            rsi = 100
        elif len(pos_array) == 0 and len(neg_array) > 0:
            rsi = 0
        else:
            rsi = 100 - 100/(1 + float(pos_mean)/float(neg_mean))
        return rsi

    desired_column = (column + '_' + security).lower()
    sec_df = get_security_data(security, start_date, end_date)
    sec_df['rsi'] = sec_df[desired_column].rolling(ndays).aggregate(_rsi_agg)

    if 'ax' in kwargs:
        ax = kwargs['ax']
        ax.plot(sec_df.index, sec_df.rsi)
        ax.axhline(y=thresholds[0], linestyle='--', c=black, alpha=0.5)
        ax.axhline(y=thresholds[1], linestyle='--', c=black, alpha=0.5)
        ax.set_ylim(0, 100)
    else:
        plt.plot(sec_df.index, sec_df.rsi)
        plt.axhline(y=thresholds[0], linestyle='--', c=black, alpha=0.5)
        plt.axhline(y=thresholds[1], linestyle='--', c=black, alpha=0.5)
        plt.ylim(0, 100)

class data_storage:
    def __init__(self):
        self.data_store_dict = {}

    def get_securities(self):
        """Returns the currently stored securities."""
        return self.data_store_dict

    def get_security_data(self, security, start_date,
                          end_date=date.today(), data_source='google',
                          return_df=True):
        sec_df = get_security_data(security, start_date, end_date,
                                   data_source)
        self.data_store_dict[security] = sec_df
        if return_df: 
            return sec_df

    def load_security_data(self, security, start_date=None, end_date=None):
        if security not in self.data_store_dict:
            raise ValueError('security not in data dictionary.')
        sec_df = self.data_store_dict[security].copy()

        if start_date is None and end_date is None: 
            return sec_df
        elif start_date is None and end_date is not None:
            return sec_df.query("index <= '{}'".format(end_date))
        elif start_date is not None and end_date is None:
            return sec_df.query("index >= '{}'".format(start_date))
        else:
            return sec_df.query("index >= '{}' & index <= '{}'"
                                    .format(start_date, end_date))

class security_portfolio:
    def __init__(self, total_cash_amt, verbose=False):
        self.start_cash_amt = total_cash_amt
        self.total_cash_amt = total_cash_amt
        self.security_dict = {}
        self.verbose = verbose
        self.trans_df = pd.DataFrame(columns=['date', 'security', 'trans_type',
                                              'security_price', 'amt',
                                              'total_cash_amt'])
            
    def buy_max_securities(self, ticker_symbol, security_price, trans_date):
        """Buy as many securities as possible with current cash."""
        amount = int(np.floor(self.total_cash_amt/security_price))
        self.buy_securities(ticker_symbol, amount, security_price, trans_date)

    def buy_securities(self, ticker_symbol, amount, security_price, trans_date):
        # Check if there is enough cash to buy the securities
        if amount * security_price <= self.total_cash_amt:
            if ticker_symbol in self.security_dict:
                # Increase number of securities
                self.security_dict[ticker_symbol] += amount
            else:
                self.security_dict[ticker_symbol] = amount
                
            # Decrease total cash amount
            start_cash_amt = self.total_cash_amt
            self.total_cash_amt -= amount * security_price
            added_row = [trans_date, ticker_symbol, 'Buy', security_price,
                         security_price * amount, self.total_cash_amt]
            self.trans_df.loc[self.trans_df.shape[0]] = added_row
            
            if self.verbose:
                print 'Bought {} shares of {} at {}.\n\tStart cash: {}.\n\tRemaining cash: {}.\n\tDate: {}'\
                      .format(amount, ticker_symbol, security_price, start_cash_amt, self.total_cash_amt, trans_date)
        else:
            raise Exception('You do not own enough cash to purchase this many securities.')
        
    def get_total_cash_amt(self):
        """Shows the total cash amount."""
        return self.total_cash_amt
            
    def sell_securities(self, ticker_symbol, amount, security_price, trans_date):
        if ticker_symbol in self.security_dict:
            num_of_security = self.security_dict[ticker_symbol]
            if amount > num_of_security:
                raise Exception('You do not have {} shares in this security. Specify lower amount, less than {}.'\
                                .format(amount, num_of_security))
            else:
                # Decrease number of securities
                self.security_dict[ticker_symbol] -= amount
                start_cash_amt = self.total_cash_amt
                # Increase total cash amount
                self.total_cash_amt += amount * security_price
                added_row = [trans_date, ticker_symbol, 'Sell', security_price,
                             security_price * amount, self.total_cash_amt]
                self.trans_df.loc[self.trans_df.shape[0]] = added_row
                if self.verbose:
                    print 'Sold {} shares of {} at {}.\n\tStart cash: {}.\n\tRemaining cash: {}.\n\tDate: {}'\
                          .format(amount, ticker_symbol, security_price, start_cash_amt, self.total_cash_amt, trans_date)
        else:
            raise Exception('You do not own shares in this security.')
            
    def sell_all_securities(self, ticker_symbol, security_price, trans_date):
        """Sell all of a given security."""
        if ticker_symbol in self.security_dict:
            amount = self.security_dict[ticker_symbol]
            self.sell_securities(ticker_symbol, amount, security_price,
                                 trans_date)
        else:
            raise Exception('You do not own shares in this security.')
            
    def show_portfolio(self):
        print 'Total Cash Amount: {}'.format(self.total_cash_amt)
        return self.security_dict
        
    def get_all_transactions(self):
        """Returns a Pandas DataFrame of all transactions."""
        return self.trans_df

    def get_avg_transaction_freq(self):
        """Returns the average transaction frequency."""
        return self.trans_df.date.diff().mean()

    def get_last_sell_cash_amt(self):
        """Get the total cash amt at the last sale. This is an indicator of
        the final cash value of the portfolio.
        """
        df = self.get_all_transactions()
        if 'Sell' in set(df.trans_type):
            return float(df[df.trans_type == 'Sell'].tail(1)['total_cash_amt'])
        else:
            # If there are no sales, return the starting cash amount
            return self.start_cash_amt

    def get_portfolio_value(self, verbose=False):
        """TODO: accomodate for middle of day, if there is no closing
        price yet. 

        Gets the total portfolio value, which is calculated by taking
        all current cash plus the value of all securities owned as of
        today's date.
        """

        def _get_last_close_values():
            """Returns the last closing values of each security in the 
            portfolio.
            """
            last_close_values = {}
            for sec in self.security_dict:
                # Gets last week worth of security data. We need to do this if
                # it's a weekend or a holiday and today's date will not return
                # anything.
                last_wk = get_security_data(sec,
                                            start_date=date.today()
                                                - relativedelta(days=7),
                                            end_date=date.today() 
                                           )
                # Gets the last close price.
                last_close_values[sec] = float(last_wk.ix[-1,
                                               'close_' + sec.lower()])
            return last_close_values

        last_close_values = _get_last_close_values()
        tot_port_value = self.get_total_cash_amt()
        for sec in self.security_dict:
            tot_port_value += self.security_dict[sec] * last_close_values[sec]

        if verbose:
            print self.get_total_cash_amt()
            print self.security_dict
            print last_close_values

        return tot_port_value

    def simulate_ma_crossover(self, securities, verbose=True):
        pass

def plot_securities(security_1_df, security_2_df, column_name):
    fig, ax0 = plt.subplots(figsize=(15, 8))
    ax0.step(security_1_df.date, security_1_df[column_name], color=blue)
    for t in ax0.get_yticklabels():
        t.set_color(blue)
    ax1 = ax0.twinx()
    ax1 = plt.step(security_2_df.date, security_2_df[column_name], color=red)
    for t in ax1.get_yticklabels():
        t.set_color(red)

def pairs_trade(df_1, df_2, column, suffixes, start_date,
                end_date=date.today(), threshold=2, cash_amt=10000):
    """Run pairs trading model.
    
    Inputs:
    df_1 - The first DataFrame of security information
    df_2 - The second DataFrame of security information
    column - The column to look at
    suffixes - The suffixes appended to the columns
    start_date - When to start training from
    end_date - When to finish training
    threshold - The threshold to surpass to trade
    cash_amt - Starting cash amount
    """

    def _get_column_names(column, suffixes):
        """Retrieves desired column name."""
        return [column + suf for suf in suffixes]

    holding = False
    join_df = pd.merge(df_1, df_2, on='date', suffixes=suffixes)

    column_names = _get_column_names(column, suffixes)

    for i in range(200, len(join_df)):
        rolling_means = [np.mean(join_df[col][i-200 : i]) for col in column_names]
        rolling_stds = [np.std(join_df[col][i-200 : i]) for col in column_names]
        norm_values = [(join_df[column_names[j]][i] - rolling_means[j])/rolling_stds[j] for j in range(2)]

    return join_df
