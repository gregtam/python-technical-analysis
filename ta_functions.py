from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from textwrap import dedent

from IPython.core.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data
import seaborn as sns

black = (0, 0, 0)
white = (1, 1, 1)
blue, green, red, purple, yellow, teal = sns.color_palette('deep')
blues = sns.color_palette('Blues', n_colors=6)[::-1]

def _listify_security(security):
    """
    If input is a string, convert it to a list. If the input is a list,
    keep it the same.
    """

    if type(security) is str:
        return [security]
    elif type(security) is list or type(security) is pd.DataFrame:
        return security

def _get_security_names(security_data):
    """Extracts the security names from the merged DataFrame."""

    _col_values = [word.split('_')[-1] for word in security_data.columns]
    return set(_col_values)

def generate_ma_columns(df, securities, col_name, ndays):
    """Create columns for moving averages and determines when there are
    crossovers.

    Inputs:
    df - The merged DataFrame of security data
    securities - The corresponding list of securities, ETFs, etc.
    col_name - Close, Open, etc.
    ndays - A list of the moving average lengths we want to generate

    Returns a pandas DataFrame
    """
    df = df.copy()
    if len(ndays) != 2:
        raise Exception('Length of ndays must be 2.')
    
    for security in securities:
        # Add moving average
        input_col = col_name + '_' + security
        for n in ndays:
            final_col = '{}_{}_{}d_ma'.format(col_name, security, n)
            df[final_col] = df[input_col].rolling(n).mean()

        short_ma = df['{}_{}_{}d_ma'.format(col_name, security, ndays[0])]
        long_ma = df['{}_{}_{}d_ma'.format(col_name, security, ndays[1])]

        ma_diff_col_name = 'ma_diff_' + security
        df[ma_diff_col_name] = short_ma - long_ma
        crossover = np.array(df[ma_diff_col_name])[:-1] * np.array(df[ma_diff_col_name][1:])
        crossover = [0] + crossover.tolist()

        crossover_col_name = 'crossover_' + security
        df[crossover_col_name] = np.sign(crossover)
        # Equal to 1 if crossed over from previous day to current day,
        # i.e., the signs of ma_diff switches
        df[crossover_col_name] = df[crossover_col_name].map({1:0, 0:0, -1:1})
        # Make a new variable signal_SECURITY which determines, based
        # off the moving average, whether to buy, sell or do nothing. We
        # buy when short_ma is larger than long_ma
        signal_col_name = 'crossover_signal_' + security
        df[signal_col_name] = (short_ma > long_ma).map({True: 'Buy', False: 'Sell'})
        # Next, we only do this when a crossover has occurred, so we
        # need to remove pre-existing signals
        df.loc[df[crossover_col_name] == 0, signal_col_name] = 'N/A'
    
    return df.dropna()

def generate_bollinger_columns(df, securities, col_name, bollinger_std,
                               bollinger_len):
    """Creates columns for bollinger bands and buy signals

    Inputs:
    df - The merged DataFrame of security data
    securities - The corresponding list of securities, ETFs, etc.
    col_name - Close, Open, etc.
    bollinger_std - The standard deviation of the bollinger bands
    bollinger_len - The number of days to use for the moving average

    Returns a pandas DataFrame
    """
    df = df.copy()

    for security in securities:
        input_col = col_name + '_' + security
        # Set column names for bollinger bands
        b_high = col_name + '_' + security + '_bollinger_high'
        b_low = col_name + '_' + security + '_bollinger_low'

        rolling_window = df[input_col].rolling(bollinger_len)
        rolling_mean = rolling_window.mean()
        rolling_std = rolling_window.std()

        df[b_high] = rolling_mean + bollinger_std * rolling_std
        df[b_low] = rolling_mean - bollinger_std * rolling_std

    return df.dropna()

class security_portfolio:
    def __init__(self, total_cash_amt, verbose=False):
        self.total_cash_amt = total_cash_amt
        self.security_dict = {}
        self.verbose = verbose
        self.trans_df = pd.DataFrame(columns=['date', 'security', 'trans_type',
                                              'amt', 'total_cash_amt'])
            
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
            added_row = [trans_date, ticker_symbol, 'Buy',
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
                added_row = [trans_date, ticker_symbol, 'Sell',
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
            # If there are no sales
            return None

    def get_portfolio_value(self, verbose=False):
        """TODO: accomodate for middle of day, if there is no closing
        price yet. 

        Gets the total portfolio value, which is calculated by taking
        all current cash plus the value of all securities owned as of
        today's date.
        """

        def _get_last_close_values():
            """Returns the last closing values of each security in
            the portfolio."""
            last_close_values = {}
            for sec in self.security_dict:
                # Gets last week worth of security data. We need to do this
                # if it's a weekend or a holiday and today's date will not
                # return anything.
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

def run_simulation(security_info, start_date, end_date=date.today(),
                   start_cash_amt=10000, data_source='google', **kwargs):
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
    kwargs - Remaining keyword arguments
    """

    security_info = _listify_security(security_info)

    securities = security_info
    security_data = get_security_data(security_info, start_date, end_date,
                                      data_source=data_source)

    return run_simulation_df(security_data, start_cash_amt, **kwargs)
    
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
                 Possible keys:'ma_crossovers', 'rsi', 'bollinger'
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

    def _run_simulation():
        """Runs through the simulation."""
        purchase_price = 0

        # For each day
        for index, row in security_data.iterrows():
            for security in securities:
                close_col_name = 'close_' + security
                if 'ma_crossovers' in indicators:
                    # If ma_diff is positive on a crossover, then it is
                    # trending upwards since 50d ma is smaller than 15d
                    # ma
                    crossover_col_name = 'crossover_' + security
                    ma_diff_col_name = 'ma_diff_' + security
                    if row[crossover_col_name] == 1:
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
                if 'bollinger_std' in indicators\
                       and 'bollinger_len' in indicators:
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

def plot_trades(sec_port):
    """Plots the trades."""

    trans_df = sec_port.get_all_transactions()
    for row in trans_df.itertuples():
        if row.trans_type == 'Buy':
            plt.axvline(row.date, linestyle='--', linewidth=3, color=red)
        elif row.trans_type == 'Sell':
            plt.axvline(row.date, linestyle='--', linewidth=3, color=green)

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
    """
    Gets all securities in securities and merges them into a
    DataFrame.
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

def plot_security(security, column, start_date, end_date=date.today(),
                  data_source='google', moving_averages=[], bollinger=False,
                  bollinger_std=2):
    """
    Plots a single security

    Inputs:
    security - The ticker symbols
    column - Open, Close, etc.
    start_date - A string representing the start date
    end_date - A string representing the end date (Default: today)
    data_source - The data source to pull data from (Default: yahoo)
    moving_averages - A list representing the moving average lengths to 
                      be used
    bollinger - Determines whether to plot the bollinger band
                (Default: False)
    bollinger_std - The standard deviation to use for the bollinger band
                    (Default: 2)
    """

    df = get_security_data(security, start_date, end_date,
                           data_source=data_source)

    desired_column = column + '_' + security.lower()

    # Generate the moving averages for the security
    for nday in moving_averages:
        ma_name = '{}_ma'.format(nday)
        df[ma_name] = df[desired_column].rolling(nday).mean()

    if bollinger > 0:
        for nday in moving_averages:
            ma_name = '{}_ma'.format(nday)

            # Get rolling standard deviation
            rolling_std = df[desired_column].rolling(moving_averages[0]).std()
            # Get multiple of standard deviation
            rolling_std *= bollinger_std

            # Define the upper and lower bollinger bands
            df['bollinger_high'] = df[ma_name] + rolling_std
            df['bollinger_low'] = df[ma_name] - rolling_std

            # Plot the upper and lower bollinger bands
            plt.plot(df.index, df.bollinger_high, c=black, linestyle='--',
                     alpha=0.5)
            plt.plot(df.index, df.bollinger_low, c=black, linestyle='--',
                     alpha=0.5)

    # Plot the security price
    plt.plot(df.index, df[desired_column], c=black)
    # Plot moving averages
    for i in np.arange(len(moving_averages)):
        nday = moving_averages[i]
        plt.plot(df.index, df['{}_ma'.format(nday)], c=blues[i], alpha=0.5)
    plt.tight_layout()

def plot_rsi(security, column, start_date, end_date=date.today(), ndays=14,
             thresholds=[20, 80]):
    """
    Plots a single security

    Inputs:
    security - The ticker symbols
    column - Open, Close, etc.
    start_date - A string representing the start date
    end_date - A string representing the end date (Default: today)
    ndays - The number of days to use as a lookback period (Default: 14)
    """

    def _rsi_agg(security_array):
        """
        Returns RSI
        """
        
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

    desired_column = column + '_' + security
    df = get_security_data(security, start_date, end_date)
    df['rsi'] = df[desired_column].rolling(ndays).aggregate(_rsi_agg)

    df.rsi.plot()
    # plt.plot(df.index, df.rsi)
    plt.axhline(y=thresholds[0], linestyle='--', c=black, alpha=0.5)
    plt.axhline(y=thresholds[1], linestyle='--', c=black, alpha=0.5)
    plt.ylim(0, 100)


    



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
    """
    Run pairs trading model.
    
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
