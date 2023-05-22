"""Tables are sequences of labeled columns."""

__all__ = [ 'Table', 'Plot', 'Figure' ]

import abc
import collections
import collections.abc
import copy
import functools
import inspect
import itertools
import numbers
import urllib.parse
import warnings

import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import pandas
import IPython

import datascience.formats as _formats
import datascience.util as _util
from datascience.util import make_array
import datascience.predicates as _predicates

import seaborn


def set_global_theme():
    seaborn.set_theme(palette='tab10')

    matplotlib.rcParams.update({
        'axes.labelsize': 16.0,
        'axes.titlesize': 18.0,
        'figure.titlesize': 'large',
        'font.size': 10.0,
        'legend.fontsize': 16,
        'legend.title_fontsize': 18.0,
        'lines.markersize': 9.0,
        'xtick.labelsize': 14.0,
        'xtick.major.size': 9.0,
        'xtick.minor.size': 6.0,
        'ytick.labelsize': 14.0,
        'ytick.major.size': 9.0,
        'ytick.minor.size': 6.0
    })

set_global_theme()

_ax_stack = []


class Table(collections.abc.MutableMapping):
    
    """A sequence of string-labeled columns."""
    plots = collections.deque(maxlen=10)

    def __init__(self, labels=None, formatter=_formats.default_formatter):
        """Create an empty table with column labels.

        >>> tiles = Table(make_array('letter', 'count', 'points'))
        >>> tiles
        letter | count | points

        Args:
            ``labels`` (list of strings): The column labels.

            ``formatter`` (Formatter): An instance of :class:`Formatter` that
                formats the columns' values.
        """
        self._columns = collections.OrderedDict()
        self._formats = dict()
        self.formatter = formatter

        labels = labels if labels is not None else []
        columns = [[] for _ in labels]

        self._num_rows = 0 if len(columns) == 0 else len(columns[0])

        # Add each column to table
        for column, label in zip(columns, labels):
            self[label] = column

        self.take = _RowTaker(self)
        self.exclude = _RowExcluder(self)

    # Deprecated
    @classmethod
    def empty(cls, labels=None):
        """Creates an empty table. Column labels are optional. [Deprecated]

        Args:
            ``labels`` (None or list): If ``None``, a table with 0
                columns is created.
                If a list, each element is a column label in a table with
                0 rows.

        Returns:
            A new instance of ``Table``.
        """
        warnings.warn("Table.empty(labels) is deprecated. Use Table(labels)", FutureWarning)
        if labels is None:
            return cls()
        values = [[] for label in labels]
        return cls(values, labels)

    # Deprecated
    @classmethod
    def from_rows(cls, rows, labels):
        """Create a table from a sequence of rows (fixed-length sequences). [Deprecated]"""
        warnings.warn("Table.from_rows is deprecated. Use Table(labels).with_rows(...)", FutureWarning)
        return cls(labels).with_rows(rows)

    @classmethod
    def from_records(cls, records):
        """Create a table from a sequence of records (dicts with fixed keys).
        
           Args:

               records: A list of dictionaries with same keys.

           Returns:

               If the list is empty, it will return an empty table.
               Otherwise, it will return a table with the dictionary's keys as the column name, and the corresponding data.
               If the dictionaries do not have identical keys, the keys of the first dictionary in the list is used.
               
           Example:
           
               >>> t = Table().from_records([
               ...     {'column1':'data1','column2':1}, 
               ...     {'column1':'data2','column2':2}, 
               ...     {'column1':'data3','column2':3}
               ... ])
               >>> t
               column1 | column2
               data1   | 1
               data2   | 2
               data3   | 3

        """
        if not records:
            return cls()
        labels = sorted(list(records[0].keys()))
        columns = [[rec[label] for rec in records] for label in labels]
        return cls().with_columns(zip(labels, columns))

    # Deprecated
    @classmethod
    def from_columns_dict(cls, columns):
        """Create a table from a mapping of column labels to column values. [Deprecated]"""
        warnings.warn("Table.from_columns_dict is deprecated. Use Table().with_columns(...)", FutureWarning)
        return cls().with_columns(columns.items())

    @classmethod
    def read_table(cls, filepath_or_buffer, *args, **vargs):
        """Read a table from a file or web address.
        
        Args:
            filepath_or_buffer -- string or file handle / StringIO; The string
                              could be a URL. Valid URL schemes include http,
                              ftp, s3, and file.
        
        Returns:
            a table read from argument
                              
        Example:
	
	>>> Table.read_table('https://www.inferentialthinking.com/data/sat2014.csv')
        State        | Participation Rate | Critical Reading | Math | Writing | Combined
        North Dakota | 2.3                | 612              | 620  | 584     | 1816
        Illinois     | 4.6                | 599              | 616  | 587     | 1802
        Iowa         | 3.1                | 605              | 611  | 578     | 1794
        South Dakota | 2.9                | 604              | 609  | 579     | 1792
        Minnesota    | 5.9                | 598              | 610  | 578     | 1786
        Michigan     | 3.8                | 593              | 610  | 581     | 1784
        Wisconsin    | 3.9                | 596              | 608  | 578     | 1782
        Missouri     | 4.2                | 595              | 597  | 579     | 1771
        Wyoming      | 3.3                | 590              | 599  | 573     | 1762
        Kansas       | 5.3                | 591              | 596  | 566     | 1753
        ... (41 rows omitted)
                
        """
        # Look for .csv at the end of the path; use "," as a separator if found
        try:
            path = urllib.parse.urlparse(filepath_or_buffer).path
            if 'data8.berkeley.edu' in filepath_or_buffer:
                raise ValueError('data8.berkeley.edu requires authentication, '
                                 'which is not supported.')
        except AttributeError:
            path = filepath_or_buffer

        try:
            if 'sep' not in vargs and path.endswith('.csv'):
                vargs['sep'] = ','
        except AttributeError:
            pass
        df = pandas.read_csv(filepath_or_buffer, *args, **vargs)
        return cls.from_df(df)

    def _with_columns(self, columns):
        """Create a table from a sequence of columns, copying column labels."""
        table = type(self)()
        for label, column in zip(self.labels, columns):
            self._add_column_and_format(table, label, column)
        return table

    def _add_column_and_format(self, table, label, column):
        """Add a column to table, copying the formatter from self."""
        label = self._as_label(label)
        table[label] = column
        if label in self._formats:
            table._formats[label] = self._formats[label]

    @classmethod
    def from_df(cls, df, keep_index=False):
        """Convert a Pandas DataFrame into a Table.
        
        Args:
        
            df -- Pandas DataFrame utilized for creation of Table
            
            `keep_index` -- keeps the index of the DataFrame 
            and turns it into a column called `index` in the new Table
            
        Returns:
           a table from Pandas Dataframe in argument
           
        Example:
        
        >>> sample_DF = pandas.DataFrame(
        ...             data = zip([1,2,3],['a','b','c'],['data1','data2','data3']),
        ...             columns = ['column1','column2','column3']
        ...             )
        
        >>> sample_DF
           column1 column2 column3
        0        1       a   data1
        1        2       b   data2
        2        3       c   data3
        
        >>> t = Table().from_df(sample_DF)
        
        >>> t
        column1 | column2 | column3
        1       | a       | data1
        2       | b       | data2
        3       | c       | data3        
       
        """
        t = cls()
        if keep_index:
            t.append_column("index", df.index.values)
        labels = df.columns
        for label in labels:
            t.append_column(label, df[label])
        return t

    @classmethod
    def from_array(cls, arr):
        """Convert a structured NumPy array into a Table.

           Args:
 
               arr -- A structured NumPy array

           Returns:

               A table with the field names as the column names and the corresponding data.
               
        Example:
        
        >>> arr = np.array([
        ...       ('A',1), ('B',2)], 
        ...       dtype=[('Name', 'U10'), ('Number', 'i4')]
        ...       )
                         
        >>> arr
        array([('A', 1), ('B', 2)], dtype=[('Name', '<U10'), ('Number', '<i4')])
        
        >>> t = Table().from_array(arr)
        
        >>> t
        Name | Number
        A    | 1
        B    | 2
        
        """
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

    #################
    # Magic Methods #
    #################

    def __getitem__(self, index_or_label):
        return self.column(index_or_label)

    def __setitem__(self, index_or_label, values):
        self.append_column(index_or_label, values)

    def __delitem__(self, index_or_label):
        label = self._as_label(index_or_label)
        del self._columns[label]
        if label in self._formats:
            del self._formats[label]

    def __len__(self):
        return len(self._columns)

    def __iter__(self):
        return iter(self.labels)

    # Deprecated
    def __getattr__(self, attr):
        """Return a method that applies to all columns or a table of attributes. [Deprecated]

        E.g., t.sum() on a Table will return a table with the sum of each column.
        """
        if self.columns and all(hasattr(c, attr) for c in self.columns):
            warnings.warn("Implicit column method lookup is deprecated.", FutureWarning)
            attrs = [getattr(c, attr) for c in self.columns]
            if all(callable(attr) for attr in attrs):
                @functools.wraps(attrs[0])
                def method(*args, **vargs):
                    """Create a table from the results of calling attrs."""
                    columns = [attr(*args, **vargs) for attr in attrs]
                    return self._with_columns(columns)
                return method
            else:
                return self._with_columns([[attr] for attr in attrs])
        else:
            msg = "'{0}' object has no attribute '{1}'".format(type(self).__name__, attr)
            raise AttributeError(msg)

    ####################
    # Accessing Values #
    ####################

    @property
    def num_rows(self):
        """
        Computes the number of rows in a table
        
        Returns:
            integer value stating number of rows

        Example:
        >>> t = Table().with_columns({
        ...     'letter': ['a', 'b', 'c', 'z'],
        ...     'count':  [  9,   3,   3,   1],
        ...     'points': [  1,   2,   2,  10],
        ... })
        >>> t.num_rows
        4
        """
        return self._num_rows

    @property
    def rows(self):
        """
        Return a view of all rows.
        
        Returns: 
            list-like Rows object that contains tuple-like Row objects

        Example:
        >>> t = Table().with_columns({
        ...     'letter': ['a', 'b', 'c', 'z'],
        ...     'count':  [  9,   3,   3,   1],
        ...     'points': [  1,   2,   2,  10],
        ... })
        >>> t.rows
        Rows(letter | count | points
        a      | 9     | 1
        b      | 3     | 2
        c      | 3     | 2
        z      | 1     | 10)
        """
        return self.Rows(self)

    def row(self, index):
        """Return a row."""
        return self.rows[index]

    @property
    def labels(self):
        """
        Return a tuple of column labels.
        
        Returns: 
            tuple of labels

        Example:
        >>> t = Table().with_columns({
        ...     'letter': ['a', 'b', 'c', 'z'],
        ...     'count':  [  9,   3,   3,   1],
        ...     'points': [  1,   2,   2,  10],
        ... })
        >>> t.labels
        ('letter', 'count', 'points')
        """
        return tuple(self._columns.keys())

    # Deprecated
    @property
    def column_labels(self):
        """Return a tuple of column labels. [Deprecated]"""
        warnings.warn("column_labels is deprecated; use labels", FutureWarning)
        return self.labels

    @property
    def num_columns(self):
        """Number of columns."""
        return len(self.labels)

    @property
    def columns(self):
        """
        Return a tuple of columns, each with the values in that column.
        
        Returns: 
            tuple of columns

        Example:
        >>> t = Table().with_columns({
        ...     'letter': ['a', 'b', 'c', 'z'],
        ...     'count':  [  9,   3,   3,   1],
        ...     'points': [  1,   2,   2,  10],
        ... })
        >>> t.columns
        (array(['a', 'b', 'c', 'z'], dtype='<U1'),
        array([9, 3, 3, 1]),
        array([ 1,  2,  2, 10]))
        """
        return tuple(self._columns.values())

    def column(self, index_or_label):
        """Return the values of a column as an array.

        table.column(label) is equivalent to table[label].

        >>> tiles = Table().with_columns(
        ...     'letter', make_array('c', 'd'),
        ...     'count',  make_array(2, 4),
        ... )

        >>> list(tiles.column('letter'))
        ['c', 'd']
        >>> tiles.column(1)
        array([2, 4])

        Args:
            label (int or str): The index or label of a column

        Returns:
            An instance of ``numpy.array``.

        Raises:
            ``ValueError``: When the ``index_or_label`` is not in the table.
        """
        if (isinstance(index_or_label, str)
                and index_or_label not in self.labels):
            raise ValueError(
                'The column "{}" is not in the table. The table contains '
                'these columns: {}'
                .format(index_or_label, ', '.join(self.labels))
            )
        if (isinstance(index_or_label, int)
                and not 0 <= index_or_label < len(self.labels)):
            raise ValueError(
                'The index {} is not in the table. Only indices between '
                '0 and {} are valid'
                .format(index_or_label, len(self.labels) - 1)
            )

        return self._columns[self._as_label(index_or_label)]

    @property
    def values(self):
        """Return data in `self` as a numpy array.

        If all columns are the same dtype, the resulting array
        will have this dtype. If there are >1 dtypes in columns,
        then the resulting array will have dtype `object`.
        """
        dtypes = [col.dtype for col in self.columns]
        if len(set(dtypes)) > 1:
            dtype = object
        else:
            dtype = None
        return np.array(self.columns, dtype=dtype).T

    def column_index(self, label):
        """
        Return the index of a column by looking up its label.
        
        Args:
            ``label`` (str) -- label value of a column

        Returns: 
            integer value specifying the index of the column label

        Example:
        >>> t = Table().with_columns({
        ...     'letter': ['a', 'b', 'c', 'z'],
        ...     'count':  [  9,   3,   3,   1],
        ...     'points': [  1,   2,   2,  10],
        ... })
        >>> t.column_index('letter')
        0
        """
        return self.labels.index(label)

    def apply(self, fn, *column_or_columns):
        """Apply ``fn`` to each element or elements of ``column_or_columns``.
        If no ``column_or_columns`` provided, `fn`` is applied to each row.

        Args:
            ``fn`` (function) -- The function to apply to each element
                of ``column_or_columns``.
            ``column_or_columns`` -- Columns containing the arguments to ``fn``
                as either column labels (``str``) or column indices (``int``).
                The number of columns must match the number of arguments
                that ``fn`` expects.

        Raises:
            ``ValueError`` -- if  ``column_label`` is not an existing
                column in the table.
            ``TypeError`` -- if insufficient number of ``column_label`` passed
                to ``fn``.

        Returns:
            An array consisting of results of applying ``fn`` to elements
            specified by ``column_label`` in each row.

        >>> t = Table().with_columns(
        ...     'letter', make_array('a', 'b', 'c', 'z'),
        ...     'count',  make_array(9, 3, 3, 1),
        ...     'points', make_array(1, 2, 2, 10))
        >>> t
        letter | count | points
        a      | 9     | 1
        b      | 3     | 2
        c      | 3     | 2
        z      | 1     | 10
        >>> t.apply(lambda x: x - 1, 'points')
        array([0, 1, 1, 9])
        >>> t.apply(lambda x, y: x * y, 'count', 'points')
        array([ 9,  6,  6, 10])
        >>> t.apply(lambda x: x - 1, 'count', 'points')
        Traceback (most recent call last):
            ...
        TypeError: <lambda>() takes 1 positional argument but 2 were given
        >>> t.apply(lambda x: x - 1, 'counts')
        Traceback (most recent call last):
            ...
        ValueError: The column "counts" is not in the table. The table contains these columns: letter, count, points

        Whole rows are passed to the function if no columns are specified.

        >>> t.apply(lambda row: row[1] * 2)
        array([18,  6,  6,  2])
        """
        if not column_or_columns:
            return np.array([fn(row) for row in self.rows])
        else:
            if len(column_or_columns) == 1 and \
                    _util.is_non_string_iterable(column_or_columns[0]):
                warnings.warn(
                   "column lists are deprecated; pass each as an argument", FutureWarning)
                column_or_columns = column_or_columns[0]
            rows = zip(*self.select(*column_or_columns).columns)
            return np.array([fn(*row) for row in rows])

    def first(self, label):
        """
        Return the zeroth item in a column.

        Args:
            ``label`` (str) -- value of column label

        Returns: 
            zeroth item of column

        Example:
        >>> t = Table().with_columns({
        ...     'letter': ['a', 'b', 'c', 'z'],
        ...     'count':  [  9,   3,   3,   1],
        ...     'points': [  1,   2,   2,  10],
        ... })
        >>> t.first('letter')
        'a'
        """
        return self.column(label)[0]

    def last(self, label):
        """
        Return the last item in a column.
        
        Args:
            ``label`` (str) -- value of column label

        Returns: 
            last item of column

        Example:
        >>> t = Table().with_columns({
        ...     'letter': ['a', 'b', 'c', 'z'],
        ...     'count':  [  9,   3,   3,   1],
        ...     'points': [  1,   2,   2,  10],
        ... })
        >>> t.last('letter')
        'z'
        """
        return self.column(label)[-1]

    ############
    # Mutation #
    ############

    def set_format(self, column_or_columns, formatter):
        """
        Set the pretty print format of a column(s) and/or convert its values.

        Args:
            ``column_or_columns``: values to group (column label or index, or array)

            ``formatter``: a function applied to a single value within the
                ``column_or_columns`` at a time or a formatter class, or formatter
                class instance.

        Returns:
            A Table with ``formatter`` applied to each ``column_or_columns``.

        The following example formats the column "balance" by passing in a formatter
        class instance. The formatter is run each time ``__repr__`` is called. So, while
        the table is formatted upon being printed to the console, the underlying values
        within the table remain untouched. It's worth noting that while ``set_format``
        returns the table with the new formatters applied, this change is applied
        directly to the original table and then the original table is returned. This
        means ``set_format`` is what's known as an inplace operation.

        >>> account_info = Table().with_columns(
        ...    "user", make_array("gfoo", "bbar", "tbaz", "hbat"),
        ...    "balance", make_array(200, 555, 125, 430))
        >>> account_info
        user | balance
        gfoo | 200
        bbar | 555
        tbaz | 125
        hbat | 430
        >>> from datascience.formats import CurrencyFormatter
        >>> account_info.set_format("balance", CurrencyFormatter("BZ$")) # Belize Dollar
        user | balance
        gfoo | BZ$200
        bbar | BZ$555
        tbaz | BZ$125
        hbat | BZ$430
        >>> account_info["balance"]
        array([200, 555, 125, 430])
        >>> account_info
        user | balance
        gfoo | BZ$200
        bbar | BZ$555
        tbaz | BZ$125
        hbat | BZ$430

        The following example formats the column "balance" by passing in a formatter
        function.

        >>> account_info = Table().with_columns(
        ...    "user", make_array("gfoo", "bbar", "tbaz", "hbat"),
        ...    "balance", make_array(200, 555, 125, 430))
        >>> account_info
        user | balance
        gfoo | 200
        bbar | 555
        tbaz | 125
        hbat | 430
        >>> def iceland_krona_formatter(value):
        ...    return f"{value} kr"
        >>> account_info.set_format("balance", iceland_krona_formatter)
        user | balance
        gfoo | 200 kr
        bbar | 555 kr
        tbaz | 125 kr
        hbat | 430 kr

        The following, formats the column "balance" by passing in a formatter class.
        Note the formatter class must have a Boolean ``converts_values`` attribute set
        and a ``format_column`` method that returns a function that formats a single
        value at a time. The ``format_column`` method accepts the name of the column and
        the value of the column as arguments and returns a formatter function that
        accepts a value and Boolean indicating whether that value is the column name.
        In the following example, if the ``if label: return value`` was removed, the
        column name "balance" would be formatted and printed as "balance kr". The
        ``converts_values`` attribute should be set to False unless a ``convert_values``
        method is also defined on the formatter class.

        >>> account_info = Table().with_columns(
        ...    "user", make_array("gfoo", "bbar", "tbaz", "hbat"),
        ...    "balance", make_array(200, 555, 125, 430))
        >>> account_info
        user | balance
        gfoo | 200
        bbar | 555
        tbaz | 125
        hbat | 430
        >>> class IcelandKronaFormatter():
        ...    def __init__(self):
        ...        self.converts_values = False
        ...
        ...    def format_column(self, label, column):
        ...        def format_krona(value, label):
        ...            if label:
        ...                return value
        ...            return f"{value} kr"
        ...
        ...        return format_krona
        >>> account_info.set_format("balance", IcelandKronaFormatter)
        user | balance
        gfoo | 200 kr
        bbar | 555 kr
        tbaz | 125 kr
        hbat | 430 kr
        >>> account_info["balance"]
        array([200, 555, 125, 430])

        ``set_format`` can also be used to convert values. If you set the
        ``converts_values`` attribute to True and define a ``convert_column`` method
        that accepts the column values and returns the converted column values on the
        formatter class, the column values will be permanently converted to the new
        column values in a one time operation.

        >>> account_info = Table().with_columns(
        ...    "user", make_array("gfoo", "bbar", "tbaz", "hbat"),
        ...    "balance", make_array(200.01, 555.55, 125.65, 430.18))
        >>> account_info
        user | balance
        gfoo | 200.01
        bbar | 555.55
        tbaz | 125.65
        hbat | 430.18
        >>> class IcelandKronaFormatter():
        ...    def __init__(self):
        ...        self.converts_values = True
        ...
        ...    def format_column(self, label, column):
        ...        def format_krona(value, label):
        ...            if label:
        ...                return value
        ...            return f"{value} kr"
        ...
        ...        return format_krona
        ...
        ...    def convert_column(self, values):
        ...        # Drop the fractional kr.
        ...        return values.astype(int)
        >>> account_info.set_format("balance", IcelandKronaFormatter)
        user | balance
        gfoo | 200 kr
        bbar | 555 kr
        tbaz | 125 kr
        hbat | 430 kr
        >>> account_info
        user | balance
        gfoo | 200 kr
        bbar | 555 kr
        tbaz | 125 kr
        hbat | 430 kr
        >>> account_info["balance"]
        array([200, 555, 125, 430])

        In the following example, multiple columns are configured to use the same
        formatter. Note the following formatter takes into account the length of all
        values in the column and formats them to be the same character length. This is
        something that the default table formatter does for you but, if you write a
        custom formatter, you must do yourself.

        >>> account_info = Table().with_columns(
        ...    "user", make_array("gfoo", "bbar", "tbaz", "hbat"),
        ...    "checking", make_array(200, 555, 125, 430),
        ...    "savings", make_array(1000, 500, 1175, 6700))
        >>> account_info
        user | checking | savings
        gfoo | 200      | 1000
        bbar | 555      | 500
        tbaz | 125      | 1175
        hbat | 430      | 6700
        >>> class IcelandKronaFormatter():
        ...    def __init__(self):
        ...        self.converts_values = False
        ...
        ...    def format_column(self, label, column):
        ...        val_width = max([len(str(v)) + len(" kr") for v in column])
        ...        val_width = max(len(str(label)), val_width)
        ...
        ...        def format_krona(value, label):
        ...            if label:
        ...                return value
        ...            return f"{value} kr".ljust(val_width)
        ...
        ...        return format_krona
        >>> account_info.set_format(["checking", "savings"], IcelandKronaFormatter)
        user | checking | savings
        gfoo | 200 kr   | 1000 kr
        bbar | 555 kr   | 500 kr
        tbaz | 125 kr   | 1175 kr
        hbat | 430 kr   | 6700 kr
        """
        if inspect.isclass(formatter):
            formatter = formatter()
        if callable(formatter) and not hasattr(formatter, 'format_column'):
            formatter = _formats.FunctionFormatter(formatter)
        if not hasattr(formatter, 'format_column'):
            raise Exception('Expected Formatter or function: ' + str(formatter))
        for label in self._as_labels(column_or_columns):
            if formatter.converts_values:
                self[label] = formatter.convert_column(self[label])
            self._formats[label] = formatter
        return self

    def move_to_start(self, column_label):
        """
        Move a column to be the first column.

        The following example moves column C to be the first column. Note, move_to_start
        not only returns the original table with the column moved but, it also moves
        the column in the original. This is what's known as an inplace operation.

        >>> table = Table().with_columns(
        ...    "A", make_array(1, 2, 3, 4),
        ...    "B", make_array("foo", "bar", "baz", "bat"),
        ...    "C", make_array('a', 'b', 'c', 'd'))
        >>> table
        A    | B    | C
        1    | foo  | a
        2    | bar  | b
        3    | baz  | c
        4    | bat  | d
        >>> table.move_to_start("C")
        C    | A    | B
        a    | 1    | foo
        b    | 2    | bar
        c    | 3    | baz
        d    | 4    | bat
        >>> table
        C    | A    | B
        a    | 1    | foo
        b    | 2    | bar
        c    | 3    | baz
        d    | 4    | bat
        """
        self._columns.move_to_end(self._as_label(column_label), last=False)
        return self

    def move_to_end(self, column_label):
        """
        Move a column to be the last column.

        The following example moves column A to be the last column. Note, move_to_end
        not only returns the original table with the column moved but, it also moves
        the column in the original. This is what's known as an inplace operation.

        >>> table = Table().with_columns(
        ...    "A", make_array(1, 2, 3, 4),
        ...    "B", make_array("foo", "bar", "baz", "bat"),
        ...    "C", make_array('a', 'b', 'c', 'd'))
        >>> table
        A    | B    | C
        1    | foo  | a
        2    | bar  | b
        3    | baz  | c
        4    | bat  | d
        >>> table.move_to_end("A")
        B    | C    | A
        foo  | a    | 1
        bar  | b    | 2
        baz  | c    | 3
        bat  | d    | 4
        >>> table
        B    | C    | A
        foo  | a    | 1
        bar  | b    | 2
        baz  | c    | 3
        bat  | d    | 4
        """
        self._columns.move_to_end(self._as_label(column_label))
        return self

    def append(self, row_or_table):
        """
        Append a row or all rows of a table in place. An appended table must have all
        columns of self.

        The following example appends a row record to the table,
        followed by appending a table having all columns of self.

        >>> table = Table().with_columns(
        ...    "A", make_array(1),
        ...    "B", make_array("foo"),
        ...    "C", make_array('a'))
        >>> table
        A    | B    | C
        1    | foo  | a
        >>> table.append([2, "bar", 'b'])
        A    | B    | C
        1    | foo  | a
        2    | bar  | b
        >>> table
        A    | B    | C
        1    | foo  | a
        2    | bar  | b
        >>> table.append(Table().with_columns(
        ...    "A", make_array(3, 4),
        ...    "B", make_array("baz", "bat"),
        ...    "C", make_array('c', 'd')))
        A    | B    | C
        1    | foo  | a
        2    | bar  | b
        3    | baz  | c
        4    | bat  | d
        >>> table
        A    | B    | C
        1    | foo  | a
        2    | bar  | b
        3    | baz  | c
        4    | bat  | d
        """
        if isinstance(row_or_table, np.ndarray):
            row_or_table = row_or_table.tolist()
        elif not row_or_table:
            return
        if isinstance(row_or_table, Table):
            t = row_or_table
            columns = list(t.select(self.labels)._columns.values())
            n = t.num_rows
        else:
            if (len(list(row_or_table)) != self.num_columns):
                raise Exception('Row should have '+ str(self.num_columns) + " columns")
            columns, n = [[value] for value in row_or_table], 1
        for i, column in enumerate(self._columns):
            if self.num_rows:
                self._columns[column] = np.append(self[column], columns[i])
            else:
                self._columns[column] = np.array(columns[i])
        self._num_rows += n
        return self

    def append_column(self, label, values, formatter=None):
        """Appends a column to the table or replaces a column.

        ``__setitem__`` is aliased to this method:
            ``table.append_column('new_col', make_array(1, 2, 3))`` is equivalent to
            ``table['new_col'] = make_array(1, 2, 3)``.

        Args:
            ``label`` (str): The label of the new column.

            ``values`` (single value or list/array): If a single value, every
                value in the new column is ``values``.

                If a list or array, the new column contains the values in
                ``values``, which must be the same length as the table.
            ``formatter`` (single formatter): Adds a formatter to the column being
                appended. No formatter added by default.

        Returns:
            Original table with new or replaced column

        Raises:
            ``ValueError``: If
                - ``label`` is not a string.
                - ``values`` is a list/array and does not have the same length
                  as the number of rows in the table.

        >>> table = Table().with_columns(
        ...     'letter', make_array('a', 'b', 'c', 'z'),
        ...     'count',  make_array(9, 3, 3, 1),
        ...     'points', make_array(1, 2, 2, 10))
        >>> table
        letter | count | points
        a      | 9     | 1
        b      | 3     | 2
        c      | 3     | 2
        z      | 1     | 10
        >>> table.append_column('new_col1', make_array(10, 20, 30, 40))
        letter | count | points | new_col1
        a      | 9     | 1      | 10
        b      | 3     | 2      | 20
        c      | 3     | 2      | 30
        z      | 1     | 10     | 40
        >>> table.append_column('new_col2', 'hello')
        letter | count | points | new_col1 | new_col2
        a      | 9     | 1      | 10       | hello
        b      | 3     | 2      | 20       | hello
        c      | 3     | 2      | 30       | hello
        z      | 1     | 10     | 40       | hello
        >>> table.append_column(123, make_array(1, 2, 3, 4))
        Traceback (most recent call last):
            ...
        ValueError: The column label must be a string, but a int was given
        >>> table.append_column('bad_col', [1, 2])
        Traceback (most recent call last):
            ...
        ValueError: Column length mismatch. New column does not have the same number of rows as table.
        """
        # TODO(sam): Allow append_column to take in a another table, copying
        # over formatter as needed.
        if not isinstance(label, str):
            raise ValueError('The column label must be a string, but a '
                '{} was given'.format(label.__class__.__name__))

        if not isinstance(values, np.ndarray):
            # Coerce a single value to a sequence
            if not _util.is_non_string_iterable(values):
                values = [values] * max(self.num_rows, 1)

            # Manually cast `values` as an object due to this: https://github.com/data-8/datascience/issues/458
            if any(_util.is_non_string_iterable(el) for el in values):
                values = np.array(tuple(values), dtype=object)
            else:
                values = np.array(tuple(values))

        if self.num_rows != 0 and len(values) != self.num_rows:
            raise ValueError('Column length mismatch. New column does not have '
                             'the same number of rows as table.')
        else:
            self._num_rows = len(values)

        self._columns[label] = values

        if (formatter != None):
            self.set_format(label, formatter)
        return self

    def relabel(self, column_label, new_label):
        """Changes the label(s) of column(s) specified by ``column_label`` to
        labels in ``new_label``.

        Args:
            ``column_label`` -- (single str or array of str) The label(s) of
                columns to be changed to ``new_label``.

            ``new_label`` -- (single str or array of str): The label name(s)
                of columns to replace ``column_label``.

        Raises:
            ``ValueError`` -- if ``column_label`` is not in table, or if
                ``column_label`` and ``new_label`` are not of equal length.
            ``TypeError`` -- if ``column_label`` and/or ``new_label`` is not
                ``str``.

        Returns:
            Original table with ``new_label`` in place of ``column_label``.

        >>> table = Table().with_columns(
        ...     'points', make_array(1, 2, 3),
        ...     'id',     make_array(12345, 123, 5123))
        >>> table.relabel('id', 'yolo')
        points | yolo
        1      | 12345
        2      | 123
        3      | 5123
        >>> table.relabel(make_array('points', 'yolo'),
        ...   make_array('red', 'blue'))
        red  | blue
        1    | 12345
        2    | 123
        3    | 5123
        >>> table.relabel(make_array('red', 'green', 'blue'),
        ...   make_array('cyan', 'magenta', 'yellow', 'key'))
        Traceback (most recent call last):
            ...
        ValueError: Invalid arguments. column_label and new_label must be of equal length.
        """
        if isinstance(column_label, numbers.Integral):
            column_label = self._as_label(column_label)
        if isinstance(column_label, str) and isinstance(new_label, str):
            column_label, new_label = [column_label], [new_label]
        if len(column_label) != len(new_label):
            raise ValueError('Invalid arguments. column_label and new_label '
                'must be of equal length.')
        old_to_new = dict(zip(column_label, new_label)) # maps old labels to new ones
        for label in column_label:
            if not (label in self.labels):
                raise ValueError('Invalid labels. Column labels must '
                'already exist in table in order to be replaced.')
        rewrite = lambda s: old_to_new[s] if s in old_to_new else s
        columns = [(rewrite(s), c) for s, c in self._columns.items()]
        self._columns = collections.OrderedDict(columns)
        for label in column_label:
            # TODO(denero) Error when old and new columns share a name
            if label in self._formats:
                formatter = self._formats.pop(label)
                self._formats[old_to_new[label]] = formatter

        return self

    def remove(self, row_or_row_indices):
        """
        Removes a row or multiple rows of a table in place (row number is 0 indexed).
        If row_or_row_indices is not int or list, no changes will be made to the table.

        The following example removes 2nd row (row_or_row_indices = 1), followed by removing 2nd
        and 3rd rows (row_or_row_indices = [1, 2]).

        >>> table = Table().with_columns(
        ...    "A", make_array(1, 2, 3, 4),
        ...    "B", make_array("foo", "bar", "baz", "bat"),
        ...    "C", make_array('a', 'b', 'c', 'd'))
        >>> table
        A    | B    | C
        1    | foo  | a
        2    | bar  | b
        3    | baz  | c
        4    | bat  | d
        >>> table.remove(1)
        A    | B    | C
        1    | foo  | a
        3    | baz  | c
        4    | bat  | d
        >>> table
        A    | B    | C
        1    | foo  | a
        3    | baz  | c
        4    | bat  | d
        >>> table.remove([1, 2])
        A    | B    | C
        1    | foo  | a
        >>> table
        A    | B    | C
        1    | foo  | a
        """
        if not row_or_row_indices and not isinstance(row_or_row_indices, int):
            return
        if isinstance(row_or_row_indices, int):
            rows_remove = [row_or_row_indices]
        else:
            rows_remove = row_or_row_indices
        for col in self._columns:
            self._columns[col] = np.array([elem for i, elem in enumerate(self[col]) if i not in rows_remove])
        self._num_rows -= len(rows_remove)
        return self


    ##################
    # Transformation #
    ##################

    def copy(self, *, shallow=False):
        """
        Return a copy of a table.

        Args:
            ``shallow``: perform a shallow copy

        Returns:
            A copy of the table.

        By default, copy performs a deep copy of the original table. This means that
        it constructs a new object and recursively inserts copies into it of the objects
        found in the original. Note in the following example, table_copy is a deep copy
        of original_table so when original_table is updated it does not change
        table_copy as it does not contain reference's to orignal_tables objects
        due to the deep copy.

        >>> value = ["foo"]
        >>> original_table = Table().with_columns(
        ...    "A", make_array(1, 2, 3),
        ...    "B", make_array(value, ["foo", "bar"], ["foo"]),
        ... )
        >>> original_table
        A    | B
        1    | ['foo']
        2    | ['foo', 'bar']
        3    | ['foo']
        >>> table_copy = original_table.copy()
        >>> table_copy
        A    | B
        1    | ['foo']
        2    | ['foo', 'bar']
        3    | ['foo']
        >>> value.append("bar")
        >>> original_table
        A    | B
        1    | ['foo', 'bar']
        2    | ['foo', 'bar']
        3    | ['foo']
        >>> table_copy
        A    | B
        1    | ['foo']
        2    | ['foo', 'bar']
        3    | ['foo']

        By contrast, when a shallow copy is performed, a new object is constructed and
        references are inserted into it to the objects found in the original. Note in
        the following example how the update to original_table  occurs in both
        table_shallow_copy and original_table because table_shallow_copy contains
        references to the original_table.

        >>> value = ["foo"]
        >>> original_table = Table().with_columns(
        ...    "A", make_array(1, 2, 3),
        ...    "B", make_array(value, ["foo", "bar"], ["foo"]),
        ... )
        >>> original_table
        A    | B
        1    | ['foo']
        2    | ['foo', 'bar']
        3    | ['foo']
        >>> table_shallow_copy = original_table.copy(shallow=True)
        >>> table_shallow_copy
        A    | B
        1    | ['foo']
        2    | ['foo', 'bar']
        3    | ['foo']
        >>> value.append("bar")
        >>> original_table
        A    | B
        1    | ['foo', 'bar']
        2    | ['foo', 'bar']
        3    | ['foo']
        >>> table_shallow_copy
        A    | B
        1    | ['foo', 'bar']
        2    | ['foo', 'bar']
        3    | ['foo']
        """
        table = type(self)()
        for label in self.labels:
            if shallow:
                column = self[label]
            else:
                column = copy.deepcopy(self[label])
            self._add_column_and_format(table, label, column)
        return table

    def select(self, *column_or_columns):
        """Return a table with only the columns in ``column_or_columns``.

        Args:
            ``column_or_columns``: Columns to select from the ``Table`` as
            either column labels (``str``) or column indices (``int``).

        Returns:
            A new instance of ``Table`` containing only selected columns.
            The columns of the new ``Table`` are in the order given in
            ``column_or_columns``.

        Raises:
            ``KeyError`` if any of ``column_or_columns`` are not in the table.

        >>> flowers = Table().with_columns(
        ...     'Number of petals', make_array(8, 34, 5),
        ...     'Name', make_array('lotus', 'sunflower', 'rose'),
        ...     'Weight', make_array(10, 5, 6)
        ... )

        >>> flowers
        Number of petals | Name      | Weight
        8                | lotus     | 10
        34               | sunflower | 5
        5                | rose      | 6

        >>> flowers.select('Number of petals', 'Weight')
        Number of petals | Weight
        8                | 10
        34               | 5
        5                | 6

        >>> flowers  # original table unchanged
        Number of petals | Name      | Weight
        8                | lotus     | 10
        34               | sunflower | 5
        5                | rose      | 6

        >>> flowers.select(0, 2)
        Number of petals | Weight
        8                | 10
        34               | 5
        5                | 6
        """
        labels = self._varargs_as_labels(column_or_columns)
        table = type(self)()
        for label in labels:
            self._add_column_and_format(table, label, np.copy(self[label]))
        return table

    # These, along with a snippet below, are necessary for Sphinx to
    # correctly load the `take` and `exclude` docstrings.  The definitions
    # will be over-ridden during class instantiation.
    def take(self):
        raise NotImplementedError()

    def exclude(self):
        raise NotImplementedError()

    def drop(self, *column_or_columns):
        """Return a Table with only columns other than selected label or
        labels.

        Args:
            ``column_or_columns`` (string or list of strings): The header
            names or indices of the columns to be dropped.

            ``column_or_columns`` must be an existing header name, or a
            valid column index.

        Returns:
            An instance of ``Table`` with given columns removed.

        >>> t = Table().with_columns(
        ...     'burgers',  make_array('cheeseburger', 'hamburger', 'veggie burger'),
        ...     'prices',   make_array(6, 5, 5),
        ...     'calories', make_array(743, 651, 582))
        >>> t
        burgers       | prices | calories
        cheeseburger  | 6      | 743
        hamburger     | 5      | 651
        veggie burger | 5      | 582
        >>> t.drop('prices')
        burgers       | calories
        cheeseburger  | 743
        hamburger     | 651
        veggie burger | 582
        >>> t.drop(['burgers', 'calories'])
        prices
        6
        5
        5
        >>> t.drop('burgers', 'calories')
        prices
        6
        5
        5
        >>> t.drop([0, 2])
        prices
        6
        5
        5
        >>> t.drop(0, 2)
        prices
        6
        5
        5
        >>> t.drop(1)
        burgers       | calories
        cheeseburger  | 743
        hamburger     | 651
        veggie burger | 582
        """
        exclude = _varargs_labels_as_list(column_or_columns)
        return self.select([c for (i, c) in enumerate(self.labels)
                            if i not in exclude and c not in exclude])

    def where(self, column_or_label, value_or_predicate=None, other=None):
        """
        Return a new ``Table`` containing rows where ``value_or_predicate``
        returns True for values in ``column_or_label``.

        Args:
            ``column_or_label``: A column of the ``Table`` either as a label
            (``str``) or an index (``int``). Can also be an array of booleans;
            only the rows where the array value is ``True`` are kept.

            ``value_or_predicate``: If a function, it is applied to every value
            in ``column_or_label``. Only the rows where ``value_or_predicate``
            returns True are kept. If a single value, only the rows where the
            values in ``column_or_label`` are equal to ``value_or_predicate``
            are kept.

            ``other``: Optional additional column label for
            ``value_or_predicate`` to make pairwise comparisons. See the
            examples below for usage. When ``other`` is supplied,
            ``value_or_predicate`` must be a callable function.

        Returns:
            If ``value_or_predicate`` is a function, returns a new ``Table``
            containing only the rows where ``value_or_predicate(val)`` is True
            for the ``val``s in ``column_or_label``.

            If ``value_or_predicate`` is a value, returns a new ``Table``
            containing only the rows where the values in ``column_or_label``
            are equal to ``value_or_predicate``.

            If ``column_or_label`` is an array of booleans, returns a new
            ``Table`` containing only the rows where ``column_or_label`` is
            ``True``.

        >>> marbles = Table().with_columns(
        ...    "Color", make_array("Red", "Green", "Blue",
        ...                        "Red", "Green", "Green"),
        ...    "Shape", make_array("Round", "Rectangular", "Rectangular",
        ...                        "Round", "Rectangular", "Round"),
        ...    "Amount", make_array(4, 6, 12, 7, 9, 2),
        ...    "Price", make_array(1.30, 1.20, 2.00, 1.75, 0, 3.00))

        >>> marbles
        Color | Shape       | Amount | Price
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.2
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 0
        Green | Round       | 2      | 3

        Use a value to select matching rows

        >>> marbles.where("Price", 1.3)
        Color | Shape | Amount | Price
        Red   | Round | 4      | 1.3

        In general, a higher order predicate function such as the functions in
        ``datascience.predicates.are`` can be used.

        >>> from datascience.predicates import are
        >>> # equivalent to previous example
        >>> marbles.where("Price", are.equal_to(1.3))
        Color | Shape | Amount | Price
        Red   | Round | 4      | 1.3

        >>> marbles.where("Price", are.above(1.5))
        Color | Shape       | Amount | Price
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Round       | 2      | 3

        Use the optional argument ``other`` to apply predicates to compare
        columns.

        >>> marbles.where("Price", are.above, "Amount")
        Color | Shape | Amount | Price
        Green | Round | 2      | 3

        >>> marbles.where("Price", are.equal_to, "Amount") # empty table
        Color | Shape | Amount | Price
        """
        column = self._get_column(column_or_label)
        if other is not None:
            assert callable(value_or_predicate), "Predicate required for 3-arg where"
            predicate = value_or_predicate
            other = self._get_column(other)
            column = [predicate(y)(x) for x, y in zip(column, other)]
        elif value_or_predicate is not None:
            if not callable(value_or_predicate):
                predicate = _predicates.are.equal_to(value_or_predicate)
            else:
                predicate = value_or_predicate
            column = [predicate(x) for x in column]
        return self.take(np.nonzero(column)[0])

    def sort(self, column_or_label, descending=False, distinct=False):
        """Return a Table of rows sorted according to the values in a column.

        Args:
            ``column_or_label``: the column whose values are used for sorting.

            ``descending``: if True, sorting will be in descending, rather than
                ascending order.

            ``distinct``: if True, repeated values in ``column_or_label`` will
                be omitted.

        Returns:
            An instance of ``Table`` containing rows sorted based on the values
            in ``column_or_label``.

        >>> marbles = Table().with_columns(
        ...    "Color", make_array("Red", "Green", "Blue", "Red", "Green", "Green"),
        ...    "Shape", make_array("Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"),
        ...    "Amount", make_array(4, 6, 12, 7, 9, 2),
        ...    "Price", make_array(1.30, 1.30, 2.00, 1.75, 1.40, 1.00))
        >>> marbles
        Color | Shape       | Amount | Price
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 1.4
        Green | Round       | 2      | 1
        >>> marbles.sort("Amount")
        Color | Shape       | Amount | Price
        Green | Round       | 2      | 1
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 1.4
        Blue  | Rectangular | 12     | 2
        >>> marbles.sort("Amount", descending = True)
        Color | Shape       | Amount | Price
        Blue  | Rectangular | 12     | 2
        Green | Rectangular | 9      | 1.4
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 6      | 1.3
        Red   | Round       | 4      | 1.3
        Green | Round       | 2      | 1
        >>> marbles.sort(3) # the Price column
        Color | Shape       | Amount | Price
        Green | Round       | 2      | 1
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Green | Rectangular | 9      | 1.4
        Red   | Round       | 7      | 1.75
        Blue  | Rectangular | 12     | 2
        >>> marbles.sort(3, distinct = True)
        Color | Shape       | Amount | Price
        Green | Round       | 2      | 1
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 9      | 1.4
        Red   | Round       | 7      | 1.75
        Blue  | Rectangular | 12     | 2
        """
        column = self._get_column(column_or_label)
        if distinct:
            _, row_numbers = np.unique(column, return_index=True)
            if descending:
                row_numbers = np.array(row_numbers[::-1])
        else:
            if descending:
                # In order to not reverse the original row order in case of ties,
                # do the following:
                # 1. Reverse the original array.
                # 2. Sort the array in ascending order.
                # 3. Invert the array indices via: len - 1 - indice.
                # 4. Reverse the array so that it is in descending order.
                column = column[::-1]
                row_numbers = np.argsort(column, axis=0, kind='mergesort')
                row_numbers = len(row_numbers) - 1 - row_numbers
                row_numbers = np.array(row_numbers[::-1])
            else:
                row_numbers = np.argsort(column, axis=0, kind='mergesort')
        assert (row_numbers < self.num_rows).all(), row_numbers
        return self.take(row_numbers)

    def group(self, column_or_label, collect=None):
        """Group rows by unique values in a column; count or aggregate others.

        Args:
            ``column_or_label``: values to group (column label or index, or array)

            ``collect``: a function applied to values in other columns for each group

        Returns:
            A Table with each row corresponding to a unique value in ``column_or_label``,
            where the first column contains the unique values from ``column_or_label``, and the
            second contains counts for each of the unique values. If ``collect`` is
            provided, a Table is returned with all original columns, each containing values
            calculated by first grouping rows according to ``column_or_label``, then applying
            ``collect`` to each set of grouped values in the other columns.

        Note:
            The grouped column will appear first in the result table. If ``collect`` does not
            accept arguments with one of the column types, that column will be empty in the resulting
            table.

        >>> marbles = Table().with_columns(
        ...    "Color", make_array("Red", "Green", "Blue", "Red", "Green", "Green"),
        ...    "Shape", make_array("Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"),
        ...    "Amount", make_array(4, 6, 12, 7, 9, 2),
        ...    "Price", make_array(1.30, 1.30, 2.00, 1.75, 1.40, 1.00))
        >>> marbles
        Color | Shape       | Amount | Price
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 1.4
        Green | Round       | 2      | 1
        >>> marbles.group("Color") # just gives counts
        Color | count
        Blue  | 1
        Green | 3
        Red   | 2
        >>> marbles.group("Color", max) # takes the max of each grouping, in each column
        Color | Shape max   | Amount max | Price max
        Blue  | Rectangular | 12         | 2
        Green | Round       | 9          | 1.4
        Red   | Round       | 7          | 1.75
        >>> marbles.group("Shape", sum) # sum doesn't make sense for strings
        Shape       | Color sum | Amount sum | Price sum
        Rectangular |           | 27         | 4.7
        Round       |           | 13         | 4.05
        """
        # Assume that a call to group with a list of labels is a call to groups
        if _util.is_non_string_iterable(column_or_label) and \
                len(column_or_label) != self._num_rows:
            return self.groups(column_or_label, collect)

        self = self.copy(shallow=True)
        collect = _zero_on_type_error(collect)

        # Remove column used for grouping
        column = self._get_column(column_or_label)
        if isinstance(column_or_label, str) or isinstance(column_or_label, numbers.Integral):
            column_label = self._as_label(column_or_label)
            del self[column_label]
        else:
            column_label = self._unused_label('group')

        # Group by column
        groups = self.index_by(column)
        keys = sorted(groups.keys())

        # Generate grouped columns
        if collect is None:
            labels = [column_label, 'count' if column_label != 'count' else self._unused_label('count')]
            columns = [keys, [len(groups[k]) for k in keys]]
        else:
            columns, labels = [], []
            for i, label in enumerate(self.labels):
                labels.append(_collected_label(collect, label))
                c = [collect(np.array([row[i] for row in groups[k]])) for k in keys]
                columns.append(c)

        grouped = type(self)().with_columns(zip(labels, columns))
        assert column_label == self._unused_label(column_label)
        grouped[column_label] = keys
        grouped.move_to_start(column_label)
        return grouped

    def groups(self, labels, collect=None):
        """Group rows by multiple columns, count or aggregate others.

        Args:
            ``labels``: list of column names (or indices) to group on

            ``collect``: a function applied to values in other columns for each group

        Returns: A Table with each row corresponding to a unique combination of values in
            the columns specified in ``labels``, where the first columns are those
            specified in ``labels``, followed by a column of counts for each of the unique
            values. If ``collect`` is provided, a Table is returned with all original
            columns, each containing values calculated by first grouping rows according to
            to values in the ``labels`` column, then applying ``collect`` to each set of
            grouped values in the other columns.

        Note:
            The grouped columns will appear first in the result table. If ``collect`` does not
            accept arguments with one of the column types, that column will be empty in the resulting
            table.

        >>> marbles = Table().with_columns(
        ...    "Color", make_array("Red", "Green", "Blue", "Red", "Green", "Green"),
        ...    "Shape", make_array("Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"),
        ...    "Amount", make_array(4, 6, 12, 7, 9, 2),
        ...    "Price", make_array(1.30, 1.30, 2.00, 1.75, 1.40, 1.00))
        >>> marbles
        Color | Shape       | Amount | Price
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 1.4
        Green | Round       | 2      | 1
        >>> marbles.groups(["Color", "Shape"])
        Color | Shape       | count
        Blue  | Rectangular | 1
        Green | Rectangular | 2
        Green | Round       | 1
        Red   | Round       | 2
        >>> marbles.groups(["Color", "Shape"], sum)
        Color | Shape       | Amount sum | Price sum
        Blue  | Rectangular | 12         | 2
        Green | Rectangular | 15         | 2.7
        Green | Round       | 2          | 1
        Red   | Round       | 11         | 3.05
        """
        # Assume that a call to groups with one label is a call to group
        if not _util.is_non_string_iterable(labels):
            return self.group(labels, collect=collect)

        collect = _zero_on_type_error(collect)
        columns = []
        labels = self._as_labels(labels)
        for label in labels:
            if label not in self.labels:
                raise ValueError("All labels must exist in the table")
            columns.append(self._get_column(label))
        grouped = self.group(list(zip(*columns)), lambda s: s)
        grouped._columns.popitem(last=False) # Discard the column of tuples

        # Flatten grouping values and move them to front
        counts = [len(v) for v in grouped[0]]
        for label in labels[::-1]:
            grouped[label] = grouped.apply(_assert_same, label)
            grouped.move_to_start(label)

        # Aggregate other values
        if collect is None:
            count = 'count' if 'count' not in labels else self._unused_label('count')
            return grouped.select(labels).with_column(count, counts)
        else:
            for label in grouped.labels:
                if label in labels:
                    continue
                column = [collect(v) for v in grouped[label]]
                del grouped[label]
                grouped[_collected_label(collect, label)] = column
            return grouped

    def pivot(self, columns, rows, values=None, collect=None, zero=None):
        """Generate a table with a column for each unique value in ``columns``,
        with rows for each unique value in ``rows``. Each row counts/aggregates
        the values that match both row and column based on ``collect``.

        Args:
            ``columns`` -- a single column label or index, (``str`` or ``int``),
                used to create new columns, based on its unique values.
            ``rows`` -- row labels or indices, (``str`` or ``int`` or list),
                used to create new rows based on it's unique values.
            ``values`` -- column label in table for use in aggregation.
                Default None.
            ``collect`` -- aggregation function, used to group ``values``
                over row-column combinations. Default None.
            ``zero`` -- zero value to use for non-existent row-column
                combinations.

        Raises:
            TypeError -- if ``collect`` is passed in and ``values`` is not,
                vice versa.

        Returns:
            New pivot table, with row-column combinations, as specified, with
            aggregated ``values`` by ``collect`` across the intersection of
            ``columns`` and ``rows``. Simple counts provided if values and
            collect are None, as default.

        >>> titanic = Table().with_columns('age', make_array(21, 44, 56, 89, 95
        ...    , 40, 80, 45), 'survival', make_array(0,0,0,1, 1, 1, 0, 1),
        ...    'gender',  make_array('M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'),
        ...    'prediction', make_array(0, 0, 1, 1, 0, 1, 0, 1))
        >>> titanic
        age  | survival | gender | prediction
        21   | 0        | M      | 0
        44   | 0        | M      | 0
        56   | 0        | M      | 1
        89   | 1        | M      | 1
        95   | 1        | F      | 0
        40   | 1        | F      | 1
        80   | 0        | F      | 0
        45   | 1        | F      | 1
        >>> titanic.pivot('survival', 'gender')
        gender | 0    | 1
        F      | 1    | 3
        M      | 3    | 1
        >>> titanic.pivot('prediction', 'gender')
        gender | 0    | 1
        F      | 2    | 2
        M      | 2    | 2
        >>> titanic.pivot('survival', 'gender', values='age', collect = np.mean)
        gender | 0       | 1
        F      | 80      | 60
        M      | 40.3333 | 89
        >>> titanic.pivot('survival', make_array('prediction', 'gender'))
        prediction | gender | 0    | 1
        0          | F      | 1    | 1
        0          | M      | 2    | 0
        1          | F      | 0    | 2
        1          | M      | 1    | 1
        >>> titanic.pivot('survival', 'gender', values = 'age')
        Traceback (most recent call last):
           ...
        TypeError: values requires collect to be specified
        >>> titanic.pivot('survival', 'gender', collect = np.mean)
        Traceback (most recent call last):
           ...
        TypeError: collect requires values to be specified
        """
        if collect is not None and values is None:
            raise TypeError('collect requires values to be specified')
        if values is not None and collect is None:
            raise TypeError('values requires collect to be specified')
        columns = self._as_label(columns)
        rows = self._as_labels(rows)
        if values is None:
            selected = self.select([columns] + rows)
        else:
            selected = self.select([columns, values] + rows)
        grouped = selected.groups([columns] + rows, collect)

        # Generate existing combinations of values from columns in rows
        rows_values = sorted(list(set(self.select(rows).rows)))
        pivoted = type(self)(rows).with_rows(rows_values)

        # Generate other columns and add them to pivoted
        by_columns = grouped.index_by(columns)
        for label in sorted(by_columns):
            tuples = [t[1:] for t in by_columns[label]] # Discard column value
            column = _fill_with_zeros(rows_values, tuples, zero)
            pivot = self._unused_label(str(label))
            pivoted[pivot] = column
        return pivoted

    def pivot_bin(self, pivot_columns, value_column, bins=None, **vargs) :
        """Form a table with columns formed by the unique tuples in pivot_columns
        containing counts per bin of the values associated with each tuple in the value_column.

        By default, bins are chosen to contain all values in the value_column. The
        following named arguments from numpy.histogram can be applied to
        specialize bin widths:

        Args:
            ``bins`` (int or sequence of scalars): If bins is an int,
                it defines the number of equal-width bins in the given range
                (10, by default). If bins is a sequence, it defines the bin
                edges, including the rightmost edge, allowing for non-uniform
                bin widths.

            ``range`` ((float, float)): The lower and upper range of
                the bins. If not provided, range contains all values in the
                table. Values outside the range are ignored.

            ``normed`` (bool): If False, the result will contain the number of
                samples in each bin. If True, the result is normalized such that
                the integral over the range is 1.
                
        Returns:
            New pivot table with unique rows of specified ``pivot_columns``, 
            populated with 0s and 1s with respect to values from ``value_column`` 
            distributed into specified ``bins`` and ``range``.
            
        Examples:
	
	>>> t = Table.from_records([
	...   {
	...    'column1':'data1',
	...    'column2':86,
	...    'column3':'b',
	...    'column4':5,
	...   },
	...   {
	...    'column1':'data2',
	...    'column2':51,
	...    'column3':'c',
	...    'column4':3,
	...   },
	...   {
	...    'column1':'data3',
	...    'column2':32,
	...    'column3':'a',
	...    'column4':6,
	...   }
	... ])
        
        >>> t
        column1 | column2 | column3 | column4
        data1   | 86      | b       | 5
        data2   | 51      | c       | 3
        data3   | 32      | a       | 6
        
        >>> t.pivot_bin(pivot_columns='column1',value_column='column2')
        bin  | data1 | data2 | data3
        32   | 0     | 0     | 1
        37.4 | 0     | 0     | 0
        42.8 | 0     | 0     | 0
        48.2 | 0     | 1     | 0
        53.6 | 0     | 0     | 0
        59   | 0     | 0     | 0
        64.4 | 0     | 0     | 0
        69.8 | 0     | 0     | 0
        75.2 | 0     | 0     | 0
        80.6 | 1     | 0     | 0
        ... (1 rows omitted)
        
        >>> t.pivot_bin(pivot_columns=['column1','column2'],value_column='column4')
        bin  | data1-86 | data2-51 | data3-32
        3    | 0        | 1        | 0
        3.3  | 0        | 0        | 0
        3.6  | 0        | 0        | 0
        3.9  | 0        | 0        | 0
        4.2  | 0        | 0        | 0
        4.5  | 0        | 0        | 0
        4.8  | 1        | 0        | 0
        5.1  | 0        | 0        | 0
        5.4  | 0        | 0        | 0
        5.7  | 0        | 0        | 1
        ... (1 rows omitted)
        
        >>> t.pivot_bin(pivot_columns='column1',value_column='column2',bins=[20,45,100])
        bin  | data1 | data2 | data3
        20   | 0     | 0     | 1
        45   | 1     | 1     | 0
        100  | 0     | 0     | 0
        
        >>> t.pivot_bin(pivot_columns='column1',value_column='column2',bins=5,range=[30,60])
        bin  | data1 | data2 | data3
        30   | 0     | 0     | 1
        36   | 0     | 0     | 0
        42   | 0     | 0     | 0
        48   | 0     | 1     | 0
        54   | 0     | 0     | 0
        60   | 0     | 0     | 0
               
        """
        pivot_columns = _as_labels(pivot_columns)
        selected = self.select(pivot_columns + [value_column])
        grouped = selected.groups(pivot_columns, collect=lambda x:x)

        # refine bins by taking a histogram over all the data
        if bins is not None:
            vargs['bins'] = bins
        _, rbins = np.histogram(self[value_column],**vargs)
        # create a table with these bins a first column and counts for each group
        vargs['bins'] = rbins
        binned = type(self)().with_column('bin',rbins)
        for group in grouped.rows:
            col_label = "-".join(map(str,group[0:-1]))
            col_vals = group[-1]
            counts,_ = np.histogram(col_vals,**vargs)
            binned[col_label] = np.append(counts,0)
        return binned

    def stack(self, key, labels=None):
        """Takes k original columns and returns two columns, with col. 1 of
        all column names and col. 2 of all associated data.
        
        Args:
            ``key``: Name of a column from table which is the basis for stacking 
                values from the table.
             
            ``labels``: List of column names which must be included in the stacked
                representation of the table. If no value is supplied for this argument,
                then the function considers all columns from the original table.
                
        Returns:
            A table whose first column consists of stacked values from column passed in
            ``key``. The second column of this returned table consists of the column names
            passed in ``labels``, whereas the final column consists of the data values
            corresponding to the respective values in the first and second columns of the
            new table.
            
        Examples:
	
	>>> t = Table.from_records([
	...   {
	...    'column1':'data1',
	...    'column2':86,
	...    'column3':'b',
	...    'column4':5,
	...   },
	...   {
	...    'column1':'data2',
	...    'column2':51,
	...    'column3':'c',
	...    'column4':3,
	...   },
	...   {
	...    'column1':'data3',
	...    'column2':32,
	...    'column3':'a',
	...    'column4':6,
	...   }
	... ])
        
        >>> t
        column1 | column2 | column3 | column4
        data1   | 86      | b       | 5
        data2   | 51      | c       | 3
        data3   | 32      | a       | 6
        
        >>> t.stack('column2')
        column2 | column  | value
        86      | column1 | data1
        86      | column3 | b
        86      | column4 | 5
        51      | column1 | data2
        51      | column3 | c
        51      | column4 | 3
        32      | column1 | data3
        32      | column3 | a
        32      | column4 | 6
        
        >>> t.stack('column2',labels=['column4','column1'])
        column2 | column  | value
        86      | column1 | data1
        86      | column4 | 5
        51      | column1 | data2
        51      | column4 | 3
        32      | column1 | data3
        32      | column4 | 6
        
        """
        rows, labels = [], labels or self.labels
        for row in self.rows:
            [rows.append((getattr(row, key), k, v)) for k, v in row.asdict().items()
             if k != key and k in labels]
        return type(self)([key, 'column', 'value']).with_rows(rows)

    def join(self, column_label, other, other_label=None):
        """Creates a new table with the columns of self and other, containing
        rows for all values of a column that appear in both tables.

        Args:
            ``column_label``:  label of column or array of labels in self that is used to
                join  rows of ``other``.
            ``other``: Table object to join with self on matching values of
                ``column_label``.

        Kwargs:
            ``other_label``: default None, assumes ``column_label``.
                Otherwise in ``other`` used to join rows.

        Returns:
            New table self joined with ``other`` by matching values in
            ``column_label`` and ``other_label``. If the resulting join is
            empty, returns None.

        >>> table = Table().with_columns('a', make_array(9, 3, 3, 1),
        ...     'b', make_array(1, 2, 2, 10),
        ...     'c', make_array(3, 4, 5, 6))
        >>> table
        a    | b    | c
        9    | 1    | 3
        3    | 2    | 4
        3    | 2    | 5
        1    | 10   | 6
        >>> table2 = Table().with_columns( 'a', make_array(9, 1, 1, 1),
        ... 'd', make_array(1, 2, 2, 10),
        ... 'e', make_array(3, 4, 5, 6))
        >>> table2
        a    | d    | e
        9    | 1    | 3
        1    | 2    | 4
        1    | 2    | 5
        1    | 10   | 6
        >>> table.join('a', table2)
        a    | b    | c    | d    | e
        1    | 10   | 6    | 2    | 4
        1    | 10   | 6    | 2    | 5
        1    | 10   | 6    | 10   | 6
        9    | 1    | 3    | 1    | 3
        >>> table.join('a', table2, 'a') # Equivalent to previous join
        a    | b    | c    | d    | e
        1    | 10   | 6    | 2    | 4
        1    | 10   | 6    | 2    | 5
        1    | 10   | 6    | 10   | 6
        9    | 1    | 3    | 1    | 3
        >>> table.join('a', table2, 'd') # Repeat column labels relabeled
        a    | b    | c    | a_2  | e
        1    | 10   | 6    | 9    | 3
        >>> table2 #table2 has three rows with a = 1
        a    | d    | e
        9    | 1    | 3
        1    | 2    | 4
        1    | 2    | 5
        1    | 10   | 6
        >>> table #table has only one row with a = 1
        a    | b    | c
        9    | 1    | 3
        3    | 2    | 4
        3    | 2    | 5
        1    | 10   | 6
        >>> table.join(['a', 'b'], table2, ['a', 'd']) # joining on multiple columns
        a    | b    | c    | e
        1    | 10   | 6    | 6
        9    | 1    | 3    | 3
        """
        if self.num_rows == 0 or other.num_rows == 0:
            return None
        if not other_label:
            other_label = column_label

        # checking to see if joining on multiple columns
        if _util.is_non_string_iterable(column_label):
            # then we are going to be joining multiple labels
            return self._multiple_join(column_label, other, other_label)

        # original single column join
        return self._join(column_label, other, other_label)

    def _join(self, column_label, other, other_label=[]):
        """joins when COLUMN_LABEL is a string"""
        if self.num_rows == 0 or other.num_rows == 0:
            return None
        if not other_label:
            other_label = column_label

        self_rows = self.index_by(column_label)
        other_rows = other.index_by(other_label)
        return self._join_helper([column_label], self_rows, other, [other_label], other_rows)

    def _multiple_join(self, column_label, other, other_label=[]):
        """joins when column_label is a non-string iterable"""
        assert len(column_label) == len(other_label), 'unequal number of columns'

        self_rows = self._multi_index(column_label)
        other_rows = other._multi_index(other_label)
        return self._join_helper(column_label, self_rows, other, other_label, other_rows)


    def _join_helper(self, column_label, self_rows, other, other_label, other_rows):
        # Gather joined rows from self_rows that have join values in other_rows
        joined_rows = []
        for v, rows in self_rows.items():
            if v in other_rows:
                joined_rows += [row + o for row in rows for o in other_rows[v]]
        if not joined_rows:
            return None

        # Build joined table
        self_labels = list(self.labels)
        other_labels = [self._unused_label(s) for s in other.labels]
        if (len(set(self_labels + other_labels)) != len(list(self_labels + other_labels))):
            other_labels = [self._unused_label_in_either_table(s, other) for s in other.labels]
        other_labels_map = dict(zip(other.labels, other_labels))
        joined = type(self)(self_labels + other_labels).with_rows(joined_rows)

        # Copy formats from both tables
        joined._formats.update(self._formats)
        for label in other._formats:
            joined._formats[other_labels_map[label]] = other._formats[label]

        # Remove redundant column, but perhaps save its formatting
        for duplicate in other_label:
            del joined[other_labels_map[duplicate]]
        for duplicate in other_label:
            if duplicate not in self._formats and duplicate in other._formats:
                joined._formats[duplicate] = other._formats[duplicate]

        for col in column_label[::-1]:
            joined = joined.move_to_start(col).sort(col)

        return joined

    def stats(self, ops=(min, max, np.median, sum)):
        """
        Compute statistics for each column and place them in a table.

        Args:
            ``ops`` -- A tuple of stat functions to use to compute stats.

        Returns:
            A ``Table`` with a prepended statistic column with the name of the
            fucntion's as the values and the calculated stats values per column.

        By default stats calculates the minimum, maximum, np.median, and sum of each
        column.

        >>> table = Table().with_columns(
        ...     'A', make_array(4, 0, 6, 5),
        ...     'B', make_array(10, 20, 17, 17),
        ...     'C', make_array(18, 13, 2, 9))
        >>> table.stats()
        statistic | A    | B    | C
        min       | 0    | 10   | 2
        max       | 6    | 20   | 18
        median    | 4.5  | 17   | 11
        sum       | 15   | 64   | 42

        Note, stats are calculated even on non-numeric columns which may lead to
        unexpected behavior or in more severe cases errors. This is why it may be best
        to eliminate non-numeric columns from the table before running stats.

        >>> table = Table().with_columns(
        ...     'B', make_array(10, 20, 17, 17),
        ...     'C', make_array("foo", "bar", "baz", "baz"))
        >>> table.stats()
        statistic | B    | C
        min       | 10   | bar
        max       | 20   | foo
        median    | 17   |
        sum       | 64   |
        >>> table.select('B').stats()
        statistic | B
        min       | 10
        max       | 20
        median    | 17
        sum       | 64

        ``ops`` can also be overridden to calculate custom stats.

        >>> table = Table().with_columns(
        ...     'A', make_array(4, 0, 6, 5),
        ...     'B', make_array(10, 20, 17, 17),
        ...     'C', make_array(18, 13, 2, 9))
        >>> def weighted_average(x):
        ...     return np.average(x, weights=[1, 0, 1.5, 1.25])
        >>> table.stats(ops=(weighted_average, np.mean, np.median, np.std))
        statistic        | A       | B       | C
        weighted_average | 5.13333 | 15.1333 | 8.6
        mean             | 3.75    | 16      | 10.5
        median           | 4.5     | 17      | 11
        std              | 2.27761 | 3.67423 | 5.85235
        """
        names = [op.__name__ for op in ops]
        ops = [_zero_on_type_error(op) for op in ops]
        columns = [[op(column) for op in ops] for column in self.columns]
        table = type(self)().with_columns(zip(self.labels, columns))
        stats = table._unused_label('statistic')
        table[stats] = names
        table.move_to_start(stats)
        return table

    def _as_label(self, index_or_label):
        """Convert index to label."""
        if isinstance(index_or_label, str):
            return index_or_label
        if isinstance(index_or_label, numbers.Integral):
            return self.labels[index_or_label]
        else:
            raise ValueError(str(index_or_label) + ' is not a label or index')

    def _as_labels(self, label_or_labels):
        """Convert single label to list and convert indices to labels."""
        return [self._as_label(s) for s in _as_labels(label_or_labels)]

    def _varargs_as_labels(self, label_list):
        """Converts a list of labels or singleton list of list of labels into
        a list of labels.  Useful when labels are passed as varargs."""
        return self._as_labels(_varargs_labels_as_list(label_list))

    def _unused_label(self, label):
        """Generate an unused label."""
        original = label
        existing = self.labels
        i = 2
        while label in existing:
            label = '{}_{}'.format(original, i)
            i += 1
        return label

    def _unused_label_in_either_table(self, label, other):
        original = label
        existing_self = self.labels
        existing_other = other.labels
        i = 2
        while label in existing_self:
            label = '{}_{}'.format(original, i)
            i += 1
            while label in existing_other:
                label = '{}_{}'.format(original, i)
                i += 1
        return label

    def _get_column(self, column_or_label):
        """Convert label to column and check column length."""
        c = column_or_label
        if isinstance(c, collections.abc.Hashable) and c in self.labels:
            return self[c]
        elif isinstance(c, numbers.Integral):
            return self[c]
        elif isinstance(c, str):
            raise ValueError('label "{}" not in labels {}'.format(c, self.labels))
        else:
            assert len(c) == self.num_rows, 'column length mismatch'
            return c

    def percentile(self, p):
        """Return a new table with one row containing the pth percentile for
        each column.

        Assumes that each column only contains one type of value.

        Returns a new table with one row and the same column labels.
        The row contains the pth percentile of the original column, where the
        pth percentile of a column is the smallest value that at at least as
        large as the p% of numbers in the column.

        >>> table = Table().with_columns(
        ...     'count',  make_array(9, 3, 3, 1),
        ...     'points', make_array(1, 2, 2, 10))
        >>> table
        count | points
        9     | 1
        3     | 2
        3     | 2
        1     | 10
        >>> table.percentile(80)
        count | points
        9     | 10
        """
        percentiles = [[_util.percentile(p, column)] for column in self.columns]
        return self._with_columns(percentiles)

    def sample(self, k=None, with_replacement=True, weights=None):
        """Return a new table where k rows are randomly sampled from the
        original table.

        Args:
            ``k`` -- specifies the number of rows (``int``) to be sampled from
               the table. Default is k equal to number of rows in the table.

            ``with_replacement`` -- (``bool``) By default True;
                Samples ``k`` rows with replacement from table, else samples
                ``k`` rows without replacement.

            ``weights`` -- Array specifying probability the ith row of the
                table is sampled. Defaults to None, which samples each row
                with equal probability. ``weights`` must be a valid probability
                distribution -- i.e. an array the length of the number of rows,
                summing to 1.

        Raises:
            ValueError -- if ``weights`` is not length equal to number of rows
                in the table; or, if ``weights`` does not sum to 1.

        Returns:
            A new instance of ``Table`` with ``k`` rows resampled.

        >>> jobs = Table().with_columns(
        ...     'job',  make_array('a', 'b', 'c', 'd'),
        ...     'wage', make_array(10, 20, 15, 8))
        >>> jobs
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> jobs.sample() # doctest: +SKIP
        job  | wage
        b    | 20
        b    | 20
        a    | 10
        d    | 8
        >>> jobs.sample(with_replacement=True) # doctest: +SKIP
        job  | wage
        d    | 8
        b    | 20
        c    | 15
        a    | 10
        >>> jobs.sample(k = 2) # doctest: +SKIP
        job  | wage
        b    | 20
        c    | 15
        >>> ws =  make_array(0.5, 0.5, 0, 0)
        >>> jobs.sample(k=2, with_replacement=True, weights=ws) # doctest: +SKIP
        job  | wage
        a    | 10
        a    | 10
        >>> jobs.sample(k=2, weights=make_array(1, 0, 1, 0))
        Traceback (most recent call last):
            ...
        ValueError: probabilities do not sum to 1
        >>> jobs.sample(k=2, weights=make_array(1, 0, 0)) # Weights must be length of table.
        Traceback (most recent call last):
            ...
        ValueError: 'a' and 'p' must have same size
        """
        n = self.num_rows
        if k is None:
            k = n
        index = np.random.choice(n, k, replace=with_replacement, p=weights)
        columns = [[c[i] for i in index] for c in self.columns]
        sample = self._with_columns(columns)
        return sample

    def shuffle(self):
        """Return a new table where all the rows are randomly shuffled from the
        original table..

        Returns:
            A new instance of ``Table`` with all ``k`` rows shuffled.
        """
        return self.sample(with_replacement=False)

    def sample_from_distribution(self, distribution, k, proportions=False):
        """Return a new table with the same number of rows and a new column.
        The values in the distribution column are define a multinomial.
        They are replaced by sample counts/proportions in the output.

        >>> sizes = Table(['size', 'count']).with_rows([
        ...     ['small', 50],
        ...     ['medium', 100],
        ...     ['big', 50],
        ... ])
        >>> sizes.sample_from_distribution('count', 1000) # doctest: +SKIP
        size   | count | count sample
        small  | 50    | 239
        medium | 100   | 496
        big    | 50    | 265
        >>> sizes.sample_from_distribution('count', 1000, True) # doctest: +SKIP
        size   | count | count sample
        small  | 50    | 0.24
        medium | 100   | 0.51
        big    | 50    | 0.25
        """
        dist = self._get_column(distribution)
        total = sum(dist)
        assert total > 0 and np.all(dist >= 0), 'Counts or a distribution required'
        dist = dist/sum(dist)
        sample = np.random.multinomial(k, dist)
        if proportions:
            sample = sample / sum(sample)
        label = self._unused_label(self._as_label(distribution) + ' sample')
        return self.with_column(label, sample)

    def split(self, k):
        """Return a tuple of two tables where the first table contains
        ``k`` rows randomly sampled and the second contains the remaining rows.

        Args:
            ``k`` (int): The number of rows randomly sampled into the first
                table. ``k`` must be between 1 and ``num_rows - 1``.

        Raises:
            ``ValueError``: ``k`` is not between 1 and ``num_rows - 1``.

        Returns:
            A tuple containing two instances of ``Table``.

        >>> jobs = Table().with_columns(
        ...     'job',  make_array('a', 'b', 'c', 'd'),
        ...     'wage', make_array(10, 20, 15, 8))
        >>> jobs
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> sample, rest = jobs.split(3)
        >>> sample # doctest: +SKIP
        job  | wage
        c    | 15
        a    | 10
        b    | 20
        >>> rest # doctest: +SKIP
        job  | wage
        d    | 8
        """
        if not 1 <= k <= self.num_rows - 1:
            raise ValueError("Invalid value of k. k must be between 1 and the"
                             "number of rows - 1")

        rows = np.random.permutation(self.num_rows)

        first = self.take(rows[:k])
        rest = self.take(rows[k:])
        for column_label in self._formats:
            first._formats[column_label] = self._formats[column_label]
            rest._formats[column_label] = self._formats[column_label]
        return first, rest


    def with_row(self, row):
        """Return a table with an additional row.

        Args:
            ``row`` (sequence): A value for each column.

        Raises:
            ``ValueError``: If the row length differs from the column count.

        >>> tiles = Table(make_array('letter', 'count', 'points'))
        >>> tiles.with_row(['c', 2, 3]).with_row(['d', 4, 2])
        letter | count | points
        c      | 2     | 3
        d      | 4     | 2
        """
        self = self.copy()
        self.append(row)
        return self

    def with_rows(self, rows):
        """Return a table with additional rows.

        Args:
            ``rows`` (sequence of sequences): Each row has a value per column.

            If ``rows`` is a 2-d array, its shape must be (_, n) for n columns.

        Raises:
            ``ValueError``: If a row length differs from the column count.

        >>> tiles = Table(make_array('letter', 'count', 'points'))
        >>> tiles.with_rows(make_array(make_array('c', 2, 3),
        ...     make_array('d', 4, 2)))
        letter | count | points
        c      | 2     | 3
        d      | 4     | 2
        """
        self = self.copy()
        self.append(self._with_columns(zip(*rows)))
        return self

    def with_column(self, label, values, formatter=None):
        """Return a new table with an additional or replaced column.

        Args:
            ``label`` (str): The column label. If an existing label is used,
                the existing column will be replaced in the new table.

            ``values`` (single value or sequence): If a single value, every
                value in the new column is ``values``. If sequence of values,
                new column takes on values in ``values``.

            ``formatter`` (single value): Specifies formatter for the new column. Defaults to no formatter.

        Raises:
            ``ValueError``: If
                - ``label`` is not a valid column name
                - if ``label`` is not of type (str)
                - ``values`` is a list/array that does not have the same
                    length as the number of rows in the table.

        Returns:
            copy of original table with new or replaced column

        >>> alphabet = Table().with_column('letter', make_array('c','d'))
        >>> alphabet = alphabet.with_column('count', make_array(2, 4))
        >>> alphabet
        letter | count
        c      | 2
        d      | 4
        >>> alphabet.with_column('permutes', make_array('a', 'g'))
        letter | count | permutes
        c      | 2     | a
        d      | 4     | g
        >>> alphabet
        letter | count
        c      | 2
        d      | 4
        >>> alphabet.with_column('count', 1)
        letter | count
        c      | 1
        d      | 1
        >>> alphabet.with_column(1, make_array(1, 2))
        Traceback (most recent call last):
            ...
        ValueError: The column label must be a string, but a int was given
        >>> alphabet.with_column('bad_col', make_array(1))
        Traceback (most recent call last):
            ...
        ValueError: Column length mismatch. New column does not have the same number of rows as table.
        """
        # Ensure that if with_column is called instead of with_columns;
        # no error is raised.

        new_table = self.copy()
        if formatter == {}:
            formatter = None
        elif isinstance(formatter, dict):
            formatter = formatter["formatter"]
        new_table.append_column(label, values, formatter)
        return new_table

    def with_columns(self, *labels_and_values, **formatter):
        """Return a table with additional or replaced columns.


        Args:
            ``labels_and_values``: An alternating list of labels and values
                or a list of label-value pairs. If one of the labels is in
                existing table, then every value in the corresponding column is
                set to that value. If label has only a single value (``int``),
                every row of corresponding column takes on that value.
            ''formatter'' (single Formatter value): A single formatter value
                that will be applied to all columns being added using this
                function call.

        Raises:
            ``ValueError``: If
                - any label in ``labels_and_values`` is not a valid column
                    name, i.e if label is not of type (str).
                - if any value in ``labels_and_values`` is a list/array and
                    does not have the same length as the number of rows in the
                    table.
            ``AssertionError``:
                - 'incorrect columns format', if passed more than one sequence
                    (iterables) for  ``labels_and_values``.
                - 'even length sequence required' if missing a pair in
                    label-value pairs.


        Returns:
            Copy of original table with new or replaced columns. Columns added
            in order of labels. Equivalent to ``with_column(label, value)``
            when passed only one label-value pair.


        >>> players = Table().with_columns('player_id',
        ...     make_array(110234, 110235), 'wOBA', make_array(.354, .236))
        >>> players
        player_id | wOBA
        110234    | 0.354
        110235    | 0.236
        >>> players = players.with_columns('salaries', 'N/A', 'season', 2016)
        >>> players
        player_id | wOBA  | salaries | season
        110234    | 0.354 | N/A      | 2016
        110235    | 0.236 | N/A      | 2016
        >>> salaries = Table().with_column('salary',
        ...     make_array(500000, 15500000))
        >>> players.with_columns('salaries', salaries.column('salary'),
        ...     'bonus', make_array(6, 1), formatter=_formats.CurrencyFormatter)
        player_id | wOBA  | salaries    | season | bonus
        110234    | 0.354 | $500,000    | 2016   | $6
        110235    | 0.236 | $15,500,000 | 2016   | $1
        >>> players.with_columns(2, make_array('$600,000', '$20,000,000'))
        Traceback (most recent call last):
            ...
        ValueError: The column label must be a string, but a int was given
        >>> players.with_columns('salaries', make_array('$600,000'))
        Traceback (most recent call last):
            ...
        ValueError: Column length mismatch. New column does not have the same number of rows as table.
        """
        if not isinstance(self, Table):
            raise TypeError('Use Table().with_columns() to create a new table, \
                not Table.with_columns()')
        if len(labels_and_values) == 1:
            labels_and_values = labels_and_values[0]
        if isinstance(labels_and_values, collections.abc.Mapping):
            labels_and_values = list(labels_and_values.items())
        if not isinstance(labels_and_values, collections.abc.Sequence):
            labels_and_values = list(labels_and_values)
        if not labels_and_values:
            return self
        first = labels_and_values[0]
        if not isinstance(first, str) and hasattr(first, '__iter__'):
            for pair in labels_and_values:
                assert len(pair) == 2, 'incorrect columns format'
            labels_and_values = [x for pair in labels_and_values for x in pair]
        assert len(labels_and_values) % 2 == 0, 'Even length sequence required'
        for i in range(0, len(labels_and_values), 2):
            label, values = labels_and_values[i], labels_and_values[i+1]
            self = self.with_column(label, values, formatter)
        return self



    def relabeled(self, label, new_label):
        """Return a new table with ``label`` specifying column label(s)
        replaced by corresponding ``new_label``.

        Args:
            ``label`` -- (str or array of str) The label(s) of
                columns to be changed.

            ``new_label`` -- (str or array of str): The new label(s) of
                columns to be changed. Same number of elements as label.

        Raises:
            ``ValueError`` -- if ``label`` does not exist in
                table, or if the ``label`` and ``new_label`` are not not of
                equal length. Also, raised if ``label`` and/or ``new_label``
                are not ``str``.

        Returns:
            New table with ``new_label`` in place of ``label``.

        >>> tiles = Table().with_columns('letter', make_array('c', 'd'),
        ...    'count', make_array(2, 4))
        >>> tiles
        letter | count
        c      | 2
        d      | 4
        >>> tiles.relabeled('count', 'number')
        letter | number
        c      | 2
        d      | 4
        >>> tiles  # original table unmodified
        letter | count
        c      | 2
        d      | 4
        >>> tiles.relabeled(make_array('letter', 'count'),
        ...   make_array('column1', 'column2'))
        column1 | column2
        c       | 2
        d       | 4
        >>> tiles.relabeled(make_array('letter', 'number'),
        ...  make_array('column1', 'column2'))
        Traceback (most recent call last):
            ...
        ValueError: Invalid labels. Column labels must already exist in table in order to be replaced.
        """
        copy = self.copy()
        copy.relabel(label, new_label)
        return copy

    # Deprecated
    def with_relabeling(self, *args):
        warnings.warn("with_relabeling is deprecated; use relabeled", FutureWarning)
        return self.relabeled(*args)

    def bin(self, *columns, **vargs):
        """Group values by bin and compute counts per bin by column.

        By default, bins are chosen to contain all values in all columns. The
        following named arguments from numpy.histogram can be applied to
        specialize bin widths:

        If the original table has n columns, the resulting binned table has
        n+1 columns, where column 0 contains the lower bound of each bin.

        Args:
            ``columns`` (str or int): Labels or indices of columns to be
                binned. If empty, all columns are binned.

            ``bins`` (int or sequence of scalars): If bins is an int,
                it defines the number of equal-width bins in the given range
                (10, by default). If bins is a sequence, it defines the bin
                edges, including the rightmost edge, allowing for non-uniform
                bin widths.

            ``range`` ((float, float)): The lower and upper range of
                the bins. If not provided, range contains all values in the
                table. Values outside the range are ignored.

            ``density`` (bool): If False, the result will contain the number of
                samples in each bin. If True, the result is the value of the
                probability density function at the bin, normalized such that
                the integral over the range is 1. Note that the sum of the
                histogram values will not be equal to 1 unless bins of unity
                width are chosen; it is not a probability mass function.
        """
        if columns:
            self = self.select(*columns)
        if 'normed' in vargs:
            vargs.setdefault('density', vargs.pop('normed'))
        density = vargs.get('density', False)
        tag = 'density' if density else 'count'

        cols = list(self._columns.values())
        _, bins = np.histogram(cols, **vargs)

        binned = type(self)().with_column('bin', bins)
        for label in self.labels:
            counts, _ = np.histogram(self[label], bins=bins, density=density)
            binned[label + ' ' + tag] = np.append(counts, 0)
        return binned

    def move_column(self, label, index):
        """Returns a new table with specified column moved to the specified column index.

        Args:
            ``label`` (str) A single label of column to be moved.

            ``index`` (int) A single index of column to move to.

        >>> titanic = Table().with_columns('age', make_array(21, 44, 56, 89, 95
        ...    , 40, 80, 45), 'survival', make_array(0,0,0,1, 1, 1, 0, 1),
        ...    'gender',  make_array('M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'),
        ...    'prediction', make_array(0, 0, 1, 1, 0, 1, 0, 1))
        >>> titanic
        age  | survival | gender | prediction
        21   | 0        | M      | 0
        44   | 0        | M      | 0
        56   | 0        | M      | 1
        89   | 1        | M      | 1
        95   | 1        | F      | 0
        40   | 1        | F      | 1
        80   | 0        | F      | 0
        45   | 1        | F      | 1
        >>> titanic.move_column('survival', 3)
        age  | gender | prediction | survival
        21   | M      | 0          | 0
        44   | M      | 0          | 0
        56   | M      | 1          | 0
        89   | M      | 1          | 1
        95   | F      | 0          | 1
        40   | F      | 1          | 1
        80   | F      | 0          | 0
        45   | F      | 1          | 1
        """

        table = type(self)()
        col_order = list(self._columns)
        label_idx = col_order.index(self._as_label(label))
        col_to_move = col_order.pop(label_idx)
        col_order.insert(index, col_to_move)
        for col in col_order:
            table[col] = self[col]
        return table

    
    
    ##########################
    # Cleaning               #
    ##########################
    
    
    def _clean_for_column(self, column_name, expected_type):
        """Given a table, and column name, and the type of data you expect
           in that column (eg: str, int, or float), this function returns
           a new copy of the table with the following changes:

             - Any rows with missing values in the given column are removed.
             - Any rows with values in the given column that are not of the 
               expected type are removed.

            Returns a new table and also a table with the removed rows
        """

        def has_type(T):
            """Return a function that indicates whether value can be converted
               to the type T."""
            def is_convertible(v):
                try:
                    T(v)
                    return True
                except ValueError:
                    return False

            if T == str:
                return lambda x: True
            elif T == int:
                return lambda x: type(x) == int or is_convertible(x) and float(x) == int(x)
            elif T == float:
                return lambda x: is_convertible(x)
        
        # Remove rows w/ nans.  Columns with floats are treated different, since nans are 
        #  converted in np.nan when a floating point column is created in read_table.
        if (self.column(column_name).dtype == np.float64):
            new_table = self.where(column_name, lambda x: not np.isnan(x))
            removed_rows = self.where(column_name, lambda x: np.isnan(x))
        else:
            new_table = self.where(column_name, _predicates.are.not_equal_to('nan'))
            removed_rows = self.where(column_name, _predicates.are.equal_to('nan'))

        # Remove rows w/ values that cannot be converted to the expected type.
        removed_rows.append(new_table.where(column_name, lambda x: not has_type(expected_type)(x)))
        new_table = new_table.where(column_name, has_type(expected_type))

        # Make sure all rows have values of the expected type.  (eg: convert str's to int's for an
        #   int column.)
        new_table = new_table.with_column(column_name, new_table.apply(expected_type, column_name))

        return new_table, removed_rows

    def _clean_for_columns(self, *labels_and_types):

        if len(labels_and_types) == 1:
            labels_and_types = labels_and_types[0]
        if isinstance(labels_and_types, collections.abc.Mapping):
            labels_and_types = list(labels_and_types.items())
        if not isinstance(labels_and_types, collections.abc.Sequence):
            labels_and_types = list(labels_and_types)
        if not labels_and_types:
            return self
        first = labels_and_types[0]

        if not isinstance(first, str) and hasattr(first, '__iter__'):
            for pair in labels_and_types:
                if len(pair) != 2:
                    raise ValueError('incorrect columns format -- should be a list alternating labels and types')
            labels_and_types = [x for pair in labels_and_types for x in pair]

        if len(labels_and_types) % 2 != 0:
            raise ValueError('incorrect columns format -- should be a list alternating labels and types')

        kept = self.copy()
        removed = Table(kept.labels)
        for i in range(0, len(labels_and_types), 2):
            label, expected_type = labels_and_types[i], labels_and_types[i+1]
            kept, dropped = kept._clean_for_column(label, expected_type)
            removed.append(dropped)

        return (kept, removed)

    
    def take_clean(self, *labels_and_types):
        clean, messy = self._clean_for_columns(*labels_and_types)
        return clean

    def take_messy(self, *labels_and_types):
        clean, messy = self._clean_for_columns(*labels_and_types)
        return messy


    
    ##########################
    # Exporting / Displaying #
    ##########################

    def __str__(self):
        return self.as_text(self.max_str_rows)

    __repr__ = __str__

    def _repr_html_(self):
        return self.as_html(self.max_str_rows)

    def show(self, max_rows=0):
        """Display the table.
	
    	Args:
    	    ``max_rows``: Maximum number of rows to be output by the function
    	
    	Returns:
    	    A subset of the Table with number of rows specified in ``max_rows``.
    	    First ``max_rows`` number of rows are displayed. If no value is passed
    	    for ``max_rows``, then the entire Table is returned.
    	    
    	Examples:

        >>> t = Table().with_columns(
        ...    "column1", make_array("data1", "data2", "data3"),
        ...    "column2", make_array(86, 51, 32),
        ...    "column3", make_array("b", "c", "a"),
        ...    "column4", make_array(5, 3, 6)
        ... )

            
        >>> t
        column1 | column2 | column3 | column4
        data1   | 86      | b       | 5
        data2   | 51      | c       | 3
        data3   | 32      | a       | 6
    	
    	>>> t.show()
    	<IPython.core.display.HTML object>
    	
    	>>> t.show(max_rows=2)
    	<IPython.core.display.HTML object>
    	
    	"""
        IPython.display.display(IPython.display.HTML(self.as_html(max_rows)))

    max_str_rows = 10

    @staticmethod
    def _use_html_if_available(format_fn):
        """Use the value's HTML rendering if available, overriding format_fn."""
        def format_using_as_html(v, label=False):
            if not label and hasattr(v, 'as_html'):
                return v.as_html()
            else:
                return format_fn(v, label)
        return format_using_as_html

    def _get_column_formatters(self, max_rows, as_html):
        """Return one value formatting function per column.

        Each function has the signature f(value, label=False) -> str
        """
        formats = {s: self._formats.get(s, self.formatter) for s in self.labels}
        cols = self._columns.items()
        fmts = [formats[k].format_column(k, v[:max_rows]) for k, v in cols]
        if as_html:
            fmts = list(map(type(self)._use_html_if_available, fmts))
        return fmts

    def as_text(self, max_rows=0, sep=" | "):
        """Format table as text
            
            Args:   
                max_rows(int) The maximum number of rows to be present in the converted string of table. (Optional Argument)
                sep(str) The seperator which will appear in converted string between the columns. (Optional Argument)
                
            Returns:
                String form of the table
                
                The table is just converted to a string with columns seperated by the seperator(argument- default(' | ')) and rows seperated by '\\n'
                
                Few examples of the as_text() method are as follows: 

                1.

                >>> table = Table().with_columns({'name': ['abc', 'xyz', 'uvw'], 'age': [12,14,20],'height': [5.5,6.0,5.9],})
                >>> table
                name | age  | height
                abc  | 12   | 5.5
                xyz  | 14   | 6
                uvw  | 20   | 5.9

                >>> table_astext = table.as_text()
                >>> table_astext
                'name | age  | height\\nabc  | 12   | 5.5\\nxyz  | 14   | 6\\nuvw  | 20   | 5.9'

                >>> type(table)
                <class 'datascience.tables.Table'>

                >>> type(table_astext)
                <class 'str'>
                 
                2.

                >>> sizes = Table(['size', 'count']).with_rows([     ['small', 50],     ['medium', 100],     ['big', 50], ])
                >>> sizes
                size   | count
                small  | 50
                medium | 100
                big    | 50

                >>> sizes_astext = sizes.as_text()
                >>> sizes_astext
                'size   | count\\nsmall  | 50\\nmedium | 100\\nbig    | 50'

                3. 

                >>> sizes_astext = sizes.as_text(1)
                >>> sizes_astext
                'size  | count\\nsmall | 50\\n... (2 rows omitted)'

                4.

                >>> sizes_astext = sizes.as_text(2, ' - ')
                >>> sizes_astext
                'size   - count\\nsmall  - 50\\nmedium - 100\\n... (1 rows omitted)'

        """
        if not max_rows or max_rows > self.num_rows:
            max_rows = self.num_rows
        omitted = max(0, self.num_rows - max_rows)
        labels = self._columns.keys()
        fmts = self._get_column_formatters(max_rows, False)
        rows = [[fmt(label, label=True) for fmt, label in zip(fmts, labels)]]
        for row in itertools.islice(self.rows, max_rows):
            rows.append([f(v, label=False) for v, f in zip(row, fmts)])
        lines = [sep.join(row) for row in rows]
        if omitted:
            lines.append('... ({} rows omitted)'.format(omitted))
        return '\n'.join([line.rstrip() for line in lines])

    def as_html(self, max_rows=0):
        """Format table as HTML
            
            Args:   
                max_rows(int) The maximum number of rows to be present in the converted string of table. (Optional Argument)
                
            Returns:
                String representing the HTML form of the table
                
                The table is converted to the html format of the table which can be used on a website to represent the table.
                
                Few examples of the as_html() method are as follows. 
                - These examples seem difficult for us to observe and understand since they are in html format, 
                they are useful when you want to display the table on webpages
                
                1. Simple table being converted to HTML

                >>> table = Table().with_columns({'name': ['abc', 'xyz', 'uvw'], 'age': [12,14,20],'height': [5.5,6.0,5.9],})

                >>> table
                name | age  | height
                abc  | 12   | 5.5
                xyz  | 14   | 6
                uvw  | 20   | 5.9

                >>> table_as_html = table.as_html()
                >>> table_as_html
                '<table border="1" class="dataframe">\\n    <thead>\\n        <tr>\\n            
                <th>name</th> <th>age</th> <th>height</th>\\n        
                </tr>\\n    </thead>\\n    <tbody>\\n        
                <tr>\\n            <td>abc </td> <td>12  </td> <td>5.5   </td>\\n        </tr>\\n        
                <tr>\\n            <td>xyz </td> <td>14  </td> <td>6     </td>\\n        </tr>\\n        
                <tr>\\n            <td>uvw </td> <td>20  </td> <td>5.9   </td>\\n        </tr>\\n    
                </tbody>\\n</table>'

                2. Simple table being converted to HTML with max_rows passed in

                >>> table
                name | age  | height
                abc  | 12   | 5.5
                xyz  | 14   | 6
                uvw  | 20   | 5.9

                >>> table_as_html_2 = table.as_html(max_rows = 2)
                >>> table_as_html_2
                '<table border="1" class="dataframe">\\n    <thead>\\n        <tr>\\n            
                <th>name</th> <th>age</th> <th>height</th>\\n        
                </tr>\\n    </thead>\\n    <tbody>\\n        
                <tr>\\n            <td>abc </td> <td>12  </td> <td>5.5   </td>\\n        </tr>\\n        
                <tr>\\n            <td>xyz </td> <td>14  </td> <td>6     </td>\\n        </tr>\\n    
                </tbody>\\n</table>\\n<p>... (1 rows omitted)</p>'
        """
        if not max_rows or max_rows > self.num_rows:
            max_rows = self.num_rows
        omitted = max(0, self.num_rows - max_rows)
        labels = self.labels
        lines = [
            (0, '<table border="1" class="dataframe">'),
            (1, '<thead>'),
            (2, '<tr>'),
            (3, ' '.join('<th>' + label + '</th>' for label in labels)),
            (2, '</tr>'),
            (1, '</thead>'),
            (1, '<tbody>'),
        ]
        fmts = self._get_column_formatters(max_rows, True)
        for row in itertools.islice(self.rows, max_rows):
            lines += [
                (2, '<tr>'),
                (3, ' '.join('<td>' + fmt(v, label=False) + '</td>' for
                    v, fmt in zip(row, fmts))),
                (2, '</tr>'),
            ]
        lines.append((1, '</tbody>'))
        lines.append((0, '</table>'))
        if omitted:
            lines.append((0, '<p>... ({} rows omitted)</p>'.format(omitted)))
        return '\n'.join(4 * indent * ' ' + text for indent, text in lines)

    def index_by(self, column_or_label):
        """Return a dict keyed by values in a column that contains lists of
            rows corresponding to each value.
    	
    	Args:
    	    ``columns_or_labels``: Name or label of a column of the Table,
    	    values of which are keys in the returned dict.
    	
    	Returns:
    	    A dictionary with values from the column specified in the argument
    	    ``columns_or_labels`` as keys. The corresponding data is a list of
    	    Row of values from the rest of the columns of the Table.
    	    
    	Examples:
    	
    	>>> t = Table().with_columns(
        ...    "column1", make_array("data1", "data2", "data3", "data4"),
        ...    "column2", make_array(86, 51, 32, 91),
        ...    "column3", make_array("b", "c", "a", "a"),
        ...    "column4", make_array(5, 3, 6, 9)
        ... )
            
        >>> t
        column1 | column2 | column3 | column4
        data1   | 86      | b       | 5
        data2   | 51      | c       | 3
        data3   | 32      | a       | 6
        data4   | 91      | a       | 9
    	
    	>>> t.index_by('column2')
    	{86: [Row(column1='data1', column2=86, column3='b', column4=5)], 51: [Row(column1='data2', column2=51, column3='c', column4=3)], 32: [Row(column1='data3', column2=32, column3='a', column4=6)], 91: [Row(column1='data4', column2=91, column3='a', column4=9)]}
    	
    	>>> t.index_by('column3')
    	{'b': [Row(column1='data1', column2=86, column3='b', column4=5)], 'c': [Row(column1='data2', column2=51, column3='c', column4=3)], 'a': [Row(column1='data3', column2=32, column3='a', column4=6), Row(column1='data4', column2=91, column3='a', column4=9)]}
    	    
        """
        column = self._get_column(column_or_label)
        index = {}
        for key, row in zip(column, self.rows):
            if isinstance(key, tuple):
                key_transformed = list(key)
            else:
                key_transformed = [key]
            has_null = pandas.isnull(key_transformed)
            if any(has_null):
                for i in range(len(key_transformed)):
                    if pandas.isnull(key_transformed[i]):
                        key_transformed[i] = np.nan
            key = tuple(key_transformed) if len(key_transformed) > 1 else key_transformed[0]
            index.setdefault(key, []).append(row)
        return index

    def _multi_index(self, columns_or_labels):
        """Returns a dict keyed by a tuple of the values that correspond to
        the selected COLUMNS_OR_LABELS, with values corresponding to """
        columns = [self._get_column(col) for col in columns_or_labels]
        index = {}
        for key, row in zip(zip(*columns), self.rows):
            index.setdefault(key, []).append(row)
        return index

    def to_df(self):
        """Convert the table to a Pandas DataFrame.
            
            Args:
                None
            
            Returns:
                The Pandas DataFrame of the table
                
             It just converts the table to Pandas DataFrame so that we can use 
             DataFrame instead of the table at some required places.
             
             Here's an example of using the to_df() method:
             
            >>> table = Table().with_columns({'name': ['abc', 'xyz', 'uvw'],
            ... 'age': [12,14,20],
            ... 'height': [5.5,6.0,5.9],
            ... })
                        
            >>> table
            name | age  | height
            abc  | 12   | 5.5
            xyz  | 14   | 6
            uvw  | 20   | 5.9
            
            >>> table_df = table.to_df()
            
            >>> table_df
              name  age  height
            0  abc   12     5.5
            1  xyz   14     6.0
            2  uvw   20     5.9
            
            >>> type(table)
            <class 'datascience.tables.Table'>

            >>> type(table_df)
            <class 'pandas.core.frame.DataFrame'>

        """
        return pandas.DataFrame(self._columns)

    def to_csv(self, filename):
        """Creates a CSV file with the provided filename.

        The CSV is created in such a way that if we run
        ``table.to_csv('my_table.csv')`` we can recreate the same table with
        ``Table.read_table('my_table.csv')``.

        Args:
            ``filename`` (str): The filename of the output CSV file.

        Returns:
            None, outputs a file with name ``filename``.

        >>> jobs = Table().with_columns(
        ...     'job',  make_array('a', 'b', 'c', 'd'),
        ...     'wage', make_array(10, 20, 15, 8))
        >>> jobs
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> jobs.to_csv('my_table.csv') # doctest: +SKIP
        <outputs a file called my_table.csv in the current directory>
        """
        # index=False avoids row numbers in the output
        self.to_df().to_csv(filename, index=False)

    def to_array(self):
        """Convert the table to a structured NumPy array.

        The resulting array contains a sequence of rows from the table.

        Args:
            None

        Returns:
            arr: a NumPy array

        The following is an example of calling to_array()
        >>> t = Table().with_columns([
        ... 'letter', ['a','b','c','z'],
        ... 'count', [9,3,3,1],
        ... 'points', [1,2,2,10],
        ... ])

        >>> t
        letter | count | points
        a      | 9     | 1
        b      | 3     | 2
        c      | 3     | 2
        z      | 1     | 10

        >>> example = t.to_array()

        >>> example
        array([('a', 9,  1), ('b', 3,  2), ('c', 3,  2), ('z', 1, 10)],
        dtype=[('letter', '<U1'), ('count', '<i8'), ('points', '<i8')])

        >>> example['letter']
        array(['a', 'b', 'c', 'z'],
        dtype='<U1')
        """
        dt = np.dtype(list(zip(self.labels, (c.dtype for c in self.columns))))
        arr = np.empty_like(self.columns[0], dt)
        for label in self.labels:
            arr[label] = self[label]
        return arr

    ##################
    # Visualizations #
    ##################
    
    default_plot_options = {
        'width': 6,
        'height': 6,
        'alpha': 0.8,
        'lw' : 3
    }

    default_scatter_options = {
        'width': 6,
        'height': 5,
        'alpha': 0.8,
    }    

    default_bar_options = {
        'width': 6,
        'alpha': 0.8,
    }    

    default_hist_options = {
        'width': 6,
        'height': 4,
        'alpha': 0.8,
    }        
    
    axis_properties = [
        'clip_on',
        'title',
        'xlabel',
        'xlim',
        'xscale',
        'ylabel',
        'ylim',
        'yscale'
    ]
    
    def split(self, keys_lists, kwargs):
        args_lists = [ {} for _ in range(len(keys_lists)) ]
        args_last = { }
        for key,value in kwargs.items():
            for keys,args in zip(keys_lists, args_lists):
                if key in keys:
                    args[key] = value
                    break
            else:
                args_last[key] = value
        args_lists.append(args_last)
        return args_lists
     
    def _complete_axes(self, ax, title=None, labels=[], artists=None):
        if len(labels) > 1:
            if artists == None:
                legend = ax.legend(labels, loc=2, title=title, bbox_to_anchor=(1.05, 1))                
            else:
                legend = ax.legend(artists, labels, loc=2, title=title, bbox_to_anchor=(1.05, 1))

    def plot(self, x_column, y_column=None, legend=True, **kwargs):
        """Plot line charts for the table. 
        
        Args:
            x_column (``str``): The column containing x-axis values
            y_column (``str/array``): The column(s) containing y-axis values, or None
                to plot use all other columns for y.

        Kwargs:
            width:
            height:
            
            ...
            
            kwargs: Additional arguments that get passed into `plt.plot`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
                for additional arguments that can be passed into vargs.
        Raises:
            ValueError -- Every selected column must be numerical.            

        >>> table = Table().with_columns(
        ...     'days',  make_array(0, 1, 2, 3, 4, 5),
        ...     'price', make_array(90.5, 90.00, 83.00, 95.50, 82.00, 82.00),
        ...     'projection', make_array(90.75, 82.00, 82.50, 82.50, 83.00, 82.50))
        >>> table
        days | price | projection
        0    | 90.5  | 90.75
        1    | 90    | 82
        2    | 83    | 82.5
        3    | 95.5  | 82.5
        4    | 82    | 83
        5    | 82    | 82.5
        >>> table.plot('days') # doctest: +SKIP
        <line graph with days as x-axis and lines for price and projection>
        >>> table.plot('days', 'price') # doctest: +SKIP
        <line graph with days as x-axis and line for price>
        """

        options = self.default_plot_options.copy()
        options.update(kwargs)

        x_label = self._as_label(x_column)
        sorted_by_x_column = self.sort(x_label)

        x_data, y_labels = sorted_by_x_column._split_column_and_labels(x_label)
        
        if y_column is not None:
            y_labels = sorted_by_x_column._as_labels(y_column)

        options.setdefault('xlabel', x_label)
        if len(y_labels) == 1: 
            options.setdefault('ylabel', y_labels[0])
        
        ax, colors, plot_options = self._prep_axes(y_labels, **options)
    
        for label, color in zip(y_labels, colors):
            ax.plot(x_data, sorted_by_x_column[label], color=color, **plot_options)

        _vertical_x(ax)
        
        if legend:
            self._complete_axes(ax, labels=y_labels)
        return Plot(ax)


    def _default_colors(self):
        return [matplotlib.colors.to_rgba(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        
    def _prep_axes(self, y_labels, **kwargs):

        set_global_theme()
        
        for label in y_labels:
            if not all(isinstance(x, numbers.Real) for x in self[label]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A plot cannot be drawn for this column."
                    .format(label))

        n = len(y_labels)

        colors = list(itertools.islice(itertools.cycle(self._default_colors()), n))
        width = kwargs.pop('width')
        height = kwargs.pop('height')
        
        all_kwargs = kwargs.copy()
        
        if len(_ax_stack) == 0:
            _, ax = plt.subplots(figsize=(width, height))
        else:
            ax = _ax_stack.pop()
            # so legends don't overlap adjacent if we are being given an axis...
            fig = ax.get_figure()
            ax.get_figure().set_tight_layout(True)
        ax.clear()
        
        axis_options, plot_options = self.split([self.axis_properties], all_kwargs)
        
        ax.set(**axis_options)
        

        return ax, colors, plot_options


    def barh(self, column_for_categories, select=None, legend=True, **kwargs):
        """Plot horizontal bar charts for the table.``
        
        Args:
            ``column_for_categories`` (``str``): A column containing y-axis categories
                used to create buckets for bar chart.
        
        Kwargs:
            overlay (bool): create a chart with one color per data column;
                if False, each will be displayed separately.
            show (bool): whether to show the figure if using interactive plots; if false, the 
                figure is returned instead
            vargs: Additional arguments that get passed into `plt.barh`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.barh
                for additional arguments that can be passed into vargs.
        
        Raises:
            ValueError -- Every selected except column for ``column_for_categories``
                must be numerical.
        
        Returns:
            Horizontal bar graph with buckets specified by ``column_for_categories``.
            Each plot is labeled using the values in ``column_for_categories``
            and one plot is produced for every other column (or for the columns
            designated by ``select``).
        
        >>> t = Table().with_columns(
        ...     'Furniture', make_array('chairs', 'tables', 'desks'),
        ...     'Count', make_array(6, 1, 2),
        ...     'Price', make_array(10, 20, 30)
        ...     )
        >>> t
        Furniture | Count | Price
        chairs    | 6     | 10
        tables    | 1     | 20
        desks     | 2     | 30
        >>> t.barh('Furniture') # doctest: +SKIP
        <bar graph with furniture as categories and bars for count and price>
        >>> t.barh('Furniture', 'Price') # doctest: +SKIP
        <bar graph with furniture as categories and bars for price>
        >>> t.barh('Furniture', make_array(1, 2)) # doctest: +SKIP
        <bar graph with furniture as categories and bars for count and price>
        """

        yticks, labels = self._split_column_and_labels(column_for_categories)
        if select is not None:
            labels = self._as_labels(select)
        n = len(labels)

        index = np.arange(self.num_rows)
        margin = 0.1
        bwidth = 1 - 2 * margin
        bwidth /= n

        options = self.default_bar_options.copy()

        # Matplotlib tries to center the labels, but we already handle that
        # TODO consider changing the custom centering code and using matplotlib's default
        options['align'] = 'edge'
        
        options.update(kwargs)

        # override default height for bar charts
        if 'height' not in kwargs:
            options['height'] = max(4, len(index)/2)

        options.setdefault('ylabel', self._as_label(column_for_categories))
        if n == 1: 
            options.setdefault('xlabel', labels[0])

        ax, colors, plot_options = self._prep_axes(labels, **options)
    
        for i, (label, color) in enumerate(zip(labels, colors)):
            ypos = index + margin + (1-2*margin)*(n - 1 - i) / n
            # barh plots entries in reverse order from bottom to top
            ax.barh(ypos, self[label][::-1], bwidth, color=color, **plot_options)

        ax.set_yticks(index+0.5) # Center labels on bars
        # barh plots entries in reverse order from bottom to top
        ax.set_yticklabels(yticks[::-1], stretch='ultra-condensed')      
 
        if legend:
            self._complete_axes(ax, labels=labels)
        return Plot(ax)
        
        
        

    def scatter(self, x_column, y_column=None, fit_line=False, group=None, sizes=None, legend=True, **kwargs):
        """Creates scatterplots, optionally adding a line of best fit. Redirects to ``Table#iscatter``
        if interactive plots are enabled with ``Table#interactive_plots``

        args:
            ``column_for_x`` (``str``): the column to use for the x-axis values
                and label of the scatter plots.

            ``fit_line`` (``bool``): draw a line of best fit for each set of points.

            ``group``: a column of categories to be used for coloring dots per
                each category grouping.

            ``sizes``:  a column of values to set the relative areas of dots.

            ``legend``: whether to show the legend

        kwargs:

            ``vargs``: additional arguments that get passed into `plt.scatter`.
                see http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
                for additional arguments that can be passed into vargs. these
                include: `marker` and `norm`, to name a couple.

            ``s``: size of dots. if sizes is also provided, then dots will be
                in the range 0 to 2 * s.

        Raises:
            ValueError -- Every column, ``column_for_x`` or ``select``, must be numerical

        Returns:
            Scatter plot of values of ``column_for_x`` plotted against
            values for all other columns in self. Each plot uses the values in
            `column_for_x` for horizontal positions. One plot is produced for
            all other columns in self as y (or for the columns designated by
            `select`).


        >>> table = Table().with_columns(
        ...     'x', make_array(9, 3, 3, 1),
        ...     'y', make_array(1, 2, 2, 10),
        ...     'z', make_array(3, 4, 5, 6))
        >>> table
        x    | y    | z
        9    | 1    | 3
        3    | 2    | 4
        3    | 2    | 5
        1    | 10   | 6
        >>> table.scatter('x') # doctest: +SKIP
        <scatterplot of values in y and z on x>

        >>> table.scatter('x') # doctest: +SKIP
        <scatterplot of values in y,z on x>

        >>> table.scatter('x', fit_line=True) # doctest: +SKIP
        <scatterplot of values in y and z on x with lines of best fit>
        """

        options = self.default_scatter_options.copy()
        options.update(kwargs)

        x_label = self._as_label(x_column)
        x_data, y_labels = self._split_column_and_labels(x_label)
        
        if y_column is not None:
            y_labels = self._as_labels(y_column)
            
        if sizes is not None and sizes in y_labels:
            y_labels.remove(self._as_label(sizes))
        if group is not None and group in y_labels:
            y_labels.remove(self._as_label(group))
            
        options.setdefault('xlabel', x_label)
        if len(y_labels) == 1:
            options.setdefault('ylabel', y_labels[0]) 
            
        # set up some defaults if not given
        s = options.setdefault('s', 20)
        options['height'] = kwargs.get('height', 6)
        
        ax, colors, plot_options = self._prep_axes(y_labels, **options)

        size_label = self._unused_label('size')
        if sizes is not None:
            max_size = max(self[sizes]) ** 0.5
            size = 2 * s * self[sizes] ** 0.5 / max_size                
        else:
            size = s
        self_with_size = self.with_columns(size_label, size)
        
        def scatter(x_values, y_values, size_values, color):
            opts = plot_options.copy()
            opts.setdefault('color', color)
            artist = ax.scatter(x_values, y_values, sizes = size_values, **opts)
            if fit_line:
                m, b = np.polyfit(x_values, y_values, 1)
                minx, maxx = np.min(x_values),np.max(x_values)
                ax.plot([minx,maxx],[m*minx+b,m*maxx+b], color = color, lw = 3, label='_skip')
            return artist
        
        if group is not None:
            if len(y_labels) > 1:
                raise ValueError("Grouping is not compatible with multiple y columns")
            
            group_values = sorted(np.unique(self.column(group)))
            colors = list(itertools.islice(itertools.cycle(self._default_colors()), len(group_values)))
            legend_artists = []
            for v,c in zip(group_values, colors):
                group_data = self_with_size.where(group, v)
                legend_artists += [ scatter(group_data[x_label], group_data[y_labels[0]], group_data[size_label], c) ]
            legend_labels = group_values
            legend_title = self._as_label(group)
        else:
            legend_artists = \
              [ scatter(self_with_size[x_label], self_with_size[y_label], self_with_size[size_label], c) \
                 for y_label,c in zip(y_labels, colors) ]
            legend_labels = y_labels
            legend_title = None 
              
        _vertical_x(ax)                            
        if legend:
            self._complete_axes(ax, legend_title, legend_labels, legend_artists)
        return Plot(ax)
    

        
    def hist(self, *columns, bins=None, group=None, left_end=None, right_end=None, legend=True, **kwargs):
        """Plots one histogram for each column in columns. If no column is
        specified, plot all columns.

        Kwargs:
            bins (list or int): Lower bound for each bin in the
                histogram or number of bins. If None, bins will
                be chosen automatically.

            group (column name or index): A column of categories.  The rows are
                grouped by the values in this column, and a separate histogram is
                generated for each group.  The histograms are overlaid or plotted
                separately depending on the overlay argument.  If None, no such
                grouping is done.

            left_end (int or float) and right_end (int or float): (Not supported
                for overlayed histograms) The left and right edges of the shading of
                the histogram. If only one of these is None, then that property
                will be treated as the extreme edge of the histogram. If both are
                left None, then no shading will occur.

            kwargs: Additional arguments that get passed into :func:plt.hist.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist
                for additional arguments that can be passed into vargs. These
                include: `range`, `normed`/`density`, `cumulative`, and
                `orientation`, to name a few.

        >>> t = Table().with_columns(
        ...     'count',  make_array(9, 3, 3, 1),
        ...     'points', make_array(1, 2, 2, 10))
        >>> t
        count | points
        9     | 1
        3     | 2
        3     | 2
        1     | 10
        >>> t.hist() # doctest: +SKIP
        <histogram of values in count>
        <histogram of values in points>

        >>> t = Table().with_columns(
        ...     'value',      make_array(101, 102, 103),
        ...     'proportion', make_array(0.25, 0.5, 0.25))
        >>> t.hist(bin_column='value') # doctest: +SKIP
        <histogram of values weighted by corresponding proportions>

        >>> t = Table().with_columns(
        ...     'value',    make_array(1,   2,   3,   2,   5  ),
        ...     'category', make_array('a', 'a', 'a', 'b', 'b'))
        >>> t.hist('value', group='category') # doctest: +SKIP
        <two overlaid histograms of the data [1, 2, 3] and [2, 5]>
        """
  
        if 'unit' in kwargs:
            warnings.warn("'unit' keyword argument is ignored.")
            kwargs = kwargs.copy()
            kwargs.pop('unit')

        options = self.default_hist_options.copy()
        options.update(kwargs)
        
        if len(columns) > 0:
            labels = self._as_labels(columns)
        else:
            labels = list(self.labels)
            
        if group is not None:
            group = self._as_label(group)
            if group in labels:
                labels.remove(group)
            if len(labels) > 1:
                raise ValueError("Using group with multiple histogram value "
                                 "columns is currently unsupported.")
                
        if len(labels) == 1:
            options.setdefault('xlabel', labels[0])
        options.setdefault('ylabel', 'Percent per unit')


        # Check for non-numerical values and raise a ValueError if any found
        for label in labels:
            if any(isinstance(cell, np.flexible) for cell in self[label]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A histogram cannot be drawn for this table."
                    .format(label))

        # maybe always for group?
        if bins is not None:
            options['bins'] = bins
            
        # Use stepfilled when there are too many bins
        if (bins is not None and isinstance(bins, numbers.Integral) and bins > 76 or hasattr(bins, '__len__') and len(bins) > 76) or group is not None or len(labels) > 1:
            options['histtype'] =  'stepfilled'
            
        options['density'] = True

        values_dict = [(k, self[k]) for k in labels]    
        values_dict = collections.OrderedDict(values_dict)

        ax, colors, plot_options = self._prep_axes(labels, **options)
        _vertical_x(ax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:g}".format(100*x)))
        
        
        if left_end is not None or right_end is not None:
            if group is not None:
                raise ValueError("Using group with left_end or right_end"
                                 " is not currently unsupported.")
            if len(labels) > 1:
                raise ValueError("Using multiple value columns with left_end or right_end"
                                 " is not currently unsupported.")

            if left_end is None:
                if bins is not None and bins[0]:
                    left_end = bins[0]
                else:
                    left_end = min([min(self.column(k)) for k in labels])
            elif right_end is None:
                if bins is not None and bins[-1]:
                    right_end = bins[-1]
                else:
                    right_end = max([max(self.column(k)) for k in labels])

        if group is not None:
            if len(labels) > 1:
                raise ValueError("Grouping is not compatible with multiple histogram columns")
            
            grouped = self.group(group, np.array)
            if grouped.num_rows > 20:
                warnings.warn("It looks like you're making a grouped histogram with "
                              "a lot of groups ({:d}), which is probably incorrect."
                              .format(grouped.num_rows))
            group_values = list(grouped.column(_collected_label(np.array, labels[0])))
            colors = list(itertools.islice(itertools.cycle(self._default_colors()), len(group_values)))
            ax.hist(group_values, color=colors, **plot_options)
                
            legend_labels = grouped.column(group)
            legend_title = self._as_label(group)
        else:
            if len(labels) == 1:
                x = self[labels[0]]
            else:
                x = [self[label] for label in labels]
            heights, bins, _ = ax.hist(x, color=colors, **plot_options)

            if left_end is not None and right_end is not None:
                # only gets here if we had only one label and no group...
                x_shade, height_shade, width_shade = _compute_shading(heights, bins.copy(), left_end, right_end)
                ax.bar(x_shade, height_shade, width=width_shade,
                         color=(253/256, 204/256, 9/256), align="edge")

            legend_labels = labels
            legend_title = None
                
        if legend:
            self._complete_axes(ax, legend_title, legend_labels[::-1])
        return Plot(ax)

    def _split_column_and_labels(self, column_or_label):
        """Return the specified column and labels of other columns."""
        assert not isinstance(column_or_label, list)
        
        column = None if column_or_label is None else self._get_column(column_or_label)
        labels = [label for i, label in enumerate(self.labels) if column_or_label not in (i, label)]
        return column, labels


    def boxplot(self, **vargs):
        """Plots a boxplot for the table.

        Every column must be numerical.

        Kwargs:
            vargs: Additional arguments that get passed into `plt.boxplot`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.boxplot
                for additional arguments that can be passed into vargs. These include
                `vert` and `showmeans`.

        Returns:
            None

        Raises:
            ValueError: The Table contains columns with non-numerical values.

        >>> table = Table().with_columns(
        ...     'test1', make_array(92.5, 88, 72, 71, 99, 100, 95, 83, 94, 93),
        ...     'test2', make_array(89, 84, 74, 66, 92, 99, 88, 81, 95, 94))
        >>> table
        test1 | test2
        92.5  | 89
        88    | 84
        72    | 74
        71    | 66
        99    | 92
        100   | 99
        95    | 88
        83    | 81
        94    | 95
        93    | 94
        >>> table.boxplot() # doctest: +SKIP
        <boxplot of test1 and boxplot of test2 side-by-side on the same figure>
        """
        # Check for non-numerical values and raise a ValueError if any found
        for col in self:
            if any(isinstance(cell, np.flexible) for cell in self[col]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A histogram cannot be drawn for this table."
                    .format(col))

        columns = self._columns.copy()
        vargs['labels'] = columns.keys()
        values = list(columns.values())
        plt.boxplot(values, **vargs)


    ###########
    # Support #
    ###########

    class Row(tuple):
        _table = None  # Set by subclasses in Rows

        def __getattr__(self, column_label):
            try:
                return self[self._table.column_index(column_label)]
            except ValueError: #adding support for NumPy v1.18.0 as per changes in https://github.com/numpy/numpy/pull/14745
                raise AttributeError("Attribute ({0}) not found in row.".format(column_label))

        def item(self, index_or_label):
            """Return the item at an index or label."""
            if isinstance(index_or_label, numbers.Integral):
                index = index_or_label
            else:
                index = self._table.column_index(index_or_label)
            return self[index]

        def __repr__(self):
            return 'Row({})'.format(', '.join('{}={}'.format(
                self._table.labels[i], v.__repr__()) for i, v in enumerate(self)))

        def asdict(self):
            return collections.OrderedDict(zip(self._table.labels, self))

    class Rows(collections.abc.Sequence):
        """An iterable view over the rows in a table."""
        def __init__(self, table):
            self._table = table
            self._labels = None

        def __getitem__(self, i):
            if isinstance(i, slice):
                return (self[j] for j in range(*i.indices(len(self))))

            labels = tuple(self._table.labels)
            if labels != self._labels:
                self._labels = labels
                self._row = type('Row', (Table.Row, ), dict(_table=self._table))
            return self._row(c[i] for c in self._table._columns.values())

        def __len__(self):
            return self._table.num_rows

        def __repr__(self):
            return '{0}({1})'.format(type(self).__name__, repr(self._table))


def _is_array_integer(arr):
    """Returns True if an array contains integers (integer type or near-int
    float values) and False otherwise.

    >>> _is_array_integer(np.arange(10))
    True
    >>> _is_array_integer(np.arange(7.0, 20.0, 1.0))
    True
    >>> _is_array_integer(np.arange(0, 1, 0.1))
    False
    """
    return issubclass(arr.dtype.type, np.integer) or np.allclose(arr, np.round(arr))

def _zero_on_type_error(column_fn):
    """Wrap a function on an np.ndarray to return 0 on a type error."""
    if not column_fn:
        return column_fn
    if not callable(column_fn):
        raise TypeError('column functions must be callable')
    @functools.wraps(column_fn)
    def wrapped(column):
        try:
            return column_fn(column)
        except TypeError:
            if isinstance(column, np.ndarray):
                return column.dtype.type() # A typed zero value
            else:
                raise
    return wrapped

def _compute_shading(heights, bins, left_end, right_end):
    shade_start_idx = np.max(np.where(bins <= left_end)[0], initial=0)
    shade_end_idx = np.max(np.where(bins < right_end)[0], initial=0) + 1
    # x_shade are the bin starts, so ignore bins[-1], which is the RHS of the last bin
    x_shade = bins[:-1][shade_start_idx:shade_end_idx]
    height_shade = heights[shade_start_idx:shade_end_idx]
    width_shade = np.diff(bins[shade_start_idx:(shade_end_idx+1)])

    if left_end > x_shade[0]:
        # shrink the width by the unshaded area, then move the bin start
        width_shade[0] -= (left_end - x_shade[0])
        x_shade[0] = left_end

    original_ending = (x_shade[-1] + width_shade[-1])
    if right_end < original_ending:
        width_shade[-1] -= (original_ending - right_end)
    return x_shade, height_shade, width_shade


def _fill_with_zeros(partials, rows, zero=None):
    """Find and return values from rows for all partials. In cases where no
    row matches a partial, zero is assumed as value. For a row, the first
    (n-1) fields are assumed to be the partial, and the last field,
    the value, where n is the total number of fields in each row. It is
    assumed that there is a unique row for each partial.
    partials -- single field values or tuples of field values
    rows -- table rows
    zero -- value used when no rows match a particular partial
    """
    assert len(rows) > 0
    if not _util.is_non_string_iterable(partials):
        # Convert partials to tuple for comparison against row slice later
        partials = [(partial,) for partial in partials]

    # Construct mapping of partials to values in rows
    mapping = {}
    for row in rows:
        mapping[tuple(row[:-1])] = row[-1]

    if zero is None:
        # Try to infer zero from given row values.
        array = np.array(tuple(mapping.values()))
        if len(array.shape) == 1:
            zero = array.dtype.type()
    return np.array([mapping.get(partial, zero) for partial in partials])


def _as_labels(column_or_columns):
    """Return a list of labels for a label or labels."""
    if not _util.is_non_string_iterable(column_or_columns):
        return [column_or_columns]
    else:
        return column_or_columns

def _varargs_labels_as_list(label_list):
    """Return a list of labels for a list of labels or singleton list of list
    of labels."""
    if len(label_list) == 0:
        return []
    elif not _util.is_non_string_iterable(label_list[0]):
        # Assume everything is a label.  If not, it'll be caught later.
        return label_list
    elif len(label_list) == 1:
        return label_list[0]
    else:
        raise ValueError("Labels {} contain more than list.".format(label_list),
                         "Pass just one list of labels.")

def _assert_same(values):
    """Assert that all values are identical and return the unique value."""
    assert len(values) > 0
    first, rest = values[0], values[1:]
    for v in rest:
        assert (v == first) or (pandas.isnull(v) and pandas.isnull(first))
    return first


def _collected_label(collect, label):
    """Label of a collected column."""
    if not collect.__name__.startswith('<'):
        return label + ' ' + collect.__name__
    else:
        return label

def _vertical_x(axis, ticks=None, max_width=5):
    """Switch labels to vertical if they are long."""
    if ticks is None:
        ticks = axis.get_xticks()
        
    ticks = np.round(ticks, max(max_width + 1, 10))
        
    if (np.array(ticks) == np.rint(ticks)).all():
        ticks = np.rint(ticks).astype(np.int64)
                
    if max([len(str(tick)) for tick in ticks]) > max_width:
        axis.set_xticklabels(ticks, rotation='vertical')

###################
# Slicing support #
###################

class _RowSelector(metaclass=abc.ABCMeta):
    def __init__(self, table):
        self._table = table

    def __call__(self, row_numbers_or_slice, *args):
        if args:
            all_args = list(args)
            all_args.insert(0, row_numbers_or_slice)
            all_args = np.array(all_args)
        else:
            all_args = row_numbers_or_slice
        return self.__getitem__(all_args)

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError()


class _RowTaker(_RowSelector):
    def __getitem__(self, row_indices_or_slice):
        """Return a new Table with selected rows taken by index.

        Args:
            ``row_indices_or_slice`` (integer or array of integers):
            The row index, list of row indices or a slice of row indices to
            be selected.

        Returns:
            A new instance of ``Table`` with selected rows in order
            corresponding to ``row_indices_or_slice``.

        Raises:
            ``IndexError``, if any ``row_indices_or_slice`` is out of bounds
            with respect to column length.

        >>> grades = Table().with_columns('letter grade',
        ...     make_array('A+', 'A', 'A-', 'B+', 'B', 'B-'),
        ...     'gpa', make_array(4, 4, 3.7, 3.3, 3, 2.7))
        >>> grades
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        B+           | 3.3
        B            | 3
        B-           | 2.7
        >>> grades.take(0)
        letter grade | gpa
        A+           | 4
        >>> grades.take(-1)
        letter grade | gpa
        B-           | 2.7
        >>> grades.take(make_array(2, 1, 0))
        letter grade | gpa
        A-           | 3.7
        A            | 4
        A+           | 4
        >>> grades.take[:3]
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        >>> grades.take(np.arange(0,3))
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        >>> grades.take(0, 2)
        letter grade | gpa
        A+           | 4
        A-           | 3.7
        >>> grades.take(10)
        Traceback (most recent call last):
            ...
        IndexError: index 10 is out of bounds for axis 0 with size 6
        """
        if isinstance(row_indices_or_slice, collections.abc.Iterable):
            columns = [np.take(column, row_indices_or_slice, axis=0)
                       for column in self._table._columns.values()]
            return self._table._with_columns(columns)

        rows = self._table.rows[row_indices_or_slice]
        if isinstance(rows, Table.Row):
            rows = [rows]
        return self._table._with_columns(zip(*rows))


class _RowExcluder(_RowSelector):
    def __getitem__(self, row_indices_or_slice):
        """Return a new Table without a sequence of rows excluded by number.

        Args:
            ``row_indices_or_slice`` (integer or list of integers or slice):
                The row index, list of row indices or a slice of row indices
                to be excluded.

        Returns:
            A new instance of ``Table``.

        >>> t = Table().with_columns(
        ...     'letter grade', make_array('A+', 'A', 'A-', 'B+', 'B', 'B-'),
        ...     'gpa', make_array(4, 4, 3.7, 3.3, 3, 2.7))
        >>> t
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        B+           | 3.3
        B            | 3
        B-           | 2.7
        >>> t.exclude(4)
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        B+           | 3.3
        B-           | 2.7
        >>> t.exclude(-1)
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        B+           | 3.3
        B            | 3
        >>> t.exclude(make_array(1, 3, 4))
        letter grade | gpa
        A+           | 4
        A-           | 3.7
        B-           | 2.7
        >>> t.exclude(range(3))
        letter grade | gpa
        B+           | 3.3
        B            | 3
        B-           | 2.7
        >>> t.exclude(0, 2)
        letter grade | gpa
        A            | 4
        B+           | 3.3
        B            | 3
        B-           | 2.7

        Note that ``exclude`` also supports NumPy-like indexing and slicing:

        >>> t.exclude[:3]
        letter grade | gpa
        B+           | 3.3
        B            | 3
        B-           | 2.7

        >>> t.exclude[1, 3, 4]
        letter grade | gpa
        A+           | 4
        A-           | 3.7
        B-           | 2.7
        """
        if isinstance(row_indices_or_slice, collections.abc.Iterable):
            without_row_indices = set(row_indices_or_slice)
            rows = [row for index, row in enumerate(self._table.rows[:])
                    if index not in without_row_indices]
            return self._table._with_columns(zip(*rows))

        row_slice = row_indices_or_slice
        if not isinstance(row_slice, slice):
            row_slice %= self._table.num_rows
            row_slice = slice(row_slice, row_slice+1)
        rows = itertools.chain(self._table.rows[:row_slice.start or 0],
                               self._table.rows[row_slice.stop:])
        return self._table._with_columns(zip(*rows))

# For Sphinx: grab the docstrings from `Taker.__getitem__` and `Withouter.__getitem__`
Table.take.__doc__ = _RowTaker.__getitem__.__doc__
Table.exclude.__doc__ = _RowExcluder.__getitem__.__doc__


### Graph annotations ###

import matplotlib.patheffects as path_effects
from IPython.display import  DisplayObject

global_kwargs = {
    'clip_on' : False,
    'alpha' : 1
}

class Figure(DisplayObject):
    
    def __init__(self, fig = None):
        if fig == None:
            fig, _ = plt.subplots(1, 1, figsize=(6, 6))
        self.fig = fig
        
    def __enter__(self):
        global _ax_stack
        self.old_stack = _ax_stack
        _ax_stack = _ax_stack + list(reversed(self.fig.axes))
        
    def __exit__(self ,type, value, traceback):
        global _ax_stack
        _ax_stack = self.old_stack
        
    def __eq__(self, other): 
        if not isinstance(other, Plot):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.fig == other.fig

    def __hash__(self):
        return hash(self.fig)
    
    def _ipython_display_(self):
        """Sneaky method to avoid printing any output if this is the result of a cell"""
        pass

    @staticmethod
    def grid(nrows = 1, ncols = 1, figsize=(6, 6)):
        fig, _ = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
        return Figure(fig)
        

class Plot(DisplayObject):

    @staticmethod
    def grid(nrows = 1, ncols = 1, figsize=(6, 6)):
        fig, _ = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
        global _ax_stack
        _ax_stack = _ax_stack + list(reversed(fig.axes))

    def __init__(self, ax = None):
        global _ax_stack
        if ax == None:
            if len(_ax_stack) == 0:
                _, ax = plt.subplots(figsize=(6, 6))
            else:
                ax = _ax_stack.pop()
        self.ax = ax
        self.zorder = 30
    
    def __enter__(self):
        global _ax_stack
        self.old_stack = _ax_stack
        _ax_stack = _ax_stack + [ self.ax ]
        return self
        
    def __exit__(self ,type, value, traceback):
        global _ax_stack
        _ax_stack = self.old_stack
        
    def __eq__(self, other): 
        if not isinstance(other, Plot):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.ax == other.ax

    def __hash__(self):
        return hash(self.ax)
    
    def _ipython_display_(self):
        """Sneaky method to avoid printing any output if this is the result of a cell"""
        pass

    def set(self, property, v):
        if property in Table.axis_properties:
            self.ax.set(property, v)
            return self
        else:
            ValueError(f"Can only set these properties for plots: {Table.axis_properties}")
            
    def set_title(self, v):
        self.ax.set_title(v)
        return self

    def set_xlabel(self, v):
        self.ax.set_xlabel(v)
        return self

    def set_xlim(self, v):
        if len(v) == 2:
            v = make_array(v[0], v[1])
        else:
            v = v[0]
        self.ax.set_xlim(v)
        return self

    def set_xscale(self, v):
        self.ax.set_xscale(v)
        return self

    def set_ylabel(self, v):
        self.ax.set_ylabel(v)
        return self

    def set_ylim(self, *v):
        if len(v) == 2:
            v = make_array(v[0], v[1])
        else:
            v = v[0]
        self.ax.set_ylim(v)
        return self

    def set_yscale(self, v):
        self.ax.set_yscale(v)
        return self

    def line(self, x=None, y=None, color='blue', width=2, linestyle='solid', **kwargs):
        ax = self.ax
        self.zorder += 1

        kws = global_kwargs.copy()
        kws.update({
            'zorder' : self.zorder,
            'clip_on' : True,
            'lw' : width,
            'color' : color,
            'linestyle': linestyle
        })
        kws.update(kwargs)

        if x is None and y is None:
            raise ValueError("Must supply at least one of x and y for line")

        if y is None:
            ax.axvline(x, **kws)
        elif x is None:
            ax.axhline(y, **kws)
        else:
            if np.shape(x) == ():
                x = [ x, x ]
            if np.shape(y) == ():
                y = [ y, y ]
            ax.axline(x, y, **kws)
    
    def interval(self, *x, y = 0, color='yellow', width=10,  **kwargs):
        ax = self.ax
        self.zorder += 1

        kws = global_kwargs.copy()
        kws.update({
            'zorder' : self.zorder,
            'color' : color,
            'lw' : width,
            'solid_capstyle' : 'butt',
            'path_effects' : [path_effects.SimpleLineShadow(),
                               path_effects.Normal()]            
        })
        kws.update(kwargs)

        if np.shape(x) == (1,2):
            x = x[0]
        elif np.shape(x) != (2,):
                raise ValueError("you must provide two numbers to mark an interval")
            
        ax.plot(x, [y,y], **kws)
        
    def y_interval(self, *y, x = 0, color='yellow', width=10,  **kwargs):
        ax = self.ax
        self.zorder += 1

        kws = global_kwargs.copy()
        kws.update({
            'zorder' : self.zorder,
            'color' : color,
            'lw' : width,
            'solid_capstyle' : 'butt',
            'path_effects' : [path_effects.SimpleLineShadow(),
                               path_effects.Normal()]            
        })
        kws.update(kwargs)

        if np.shape(y) == (1,2):
            y = y[0]
        elif np.shape(y) != (2,):
                raise ValueError("you must provide two numbers to mark an interval")
            
        ax.plot([x, x], y, **kws)
    

    def dot(self, x=0, y=0, color='red', size=150, **kwargs):
        ax = self.ax
        self.zorder += 1

        kws = global_kwargs.copy()
        kws.update({
            'zorder' : self.zorder,            
            's': size,
            'marker':'o',
            'c' : color,
            'edgecolor': 'white',
            'path_effects' : [path_effects.SimplePatchShadow(),
                               path_effects.Normal()]            
        })
        kws.update(kwargs)

        if np.shape(x) != () and np.shape(y) == ():
            y = np.full(np.shape(x), y)
        elif np.shape(y) != () and np.shape(x) == ():
            x = np.full(np.shape(y), x)

        ax.scatter(x, y, **kws)

        
    def square(self, x=0, y=0, color='limegreen', size=150, **kwargs):
        self.dot(x, y, color, size, marker='s', **kwargs)