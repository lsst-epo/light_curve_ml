import sqlite3


def _connectExample():
    return sqlite3.connect('example.db')

def setupTable():
    conn = _connectExample()
    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS stocks
                 (date text, trans text, symbol text, qty real, price real)''')

    # Insert a row of data
    c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

    # Save (commit) the changes
    conn.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()


def addRows():
    conn = _connectExample()
    c = conn.cursor()

    # Larger example that inserts many records at a time
    purchases = [('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
                 ('2006-04-05', 'BUY', 'MSFT', 1000, 72.00),
                 ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
                 ]
    c.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', purchases)
    conn.commit()
    conn.close()

def safeSelect():
    conn = sqlite3.connect('example.db')
    c = conn.cursor()

    args = ("RHAT",)
    c.execute('SELECT * FROM stocks WHERE symbol=?', args)
    print(c.fetchone())

    for row in c.execute("SELECT * FROM stocks ORDER BY price"):
        print(row)


if __name__ == "__main__":
    # setupTable()
    addRows()
    safeSelect()
