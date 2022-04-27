import csv
import datetime
import numpy as np
import os
import sys

import vrnnframe as vrnnframe
from collections import defaultdict
import sqlite3

# Set data sources
data_dir = '/y/data/processed_ddos'
trace_dirs = [
    'WSUa',
    'WSUb',
    'WSUc',
    'WSUd'
]

output_dir = '/y/adravi/research/ddos_chargen_2016_processed'
output_file = '/y/adravi/research/ddos_chargen_2016_processed/ddos_chargen_2016.npz'

def get_db_connection():
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS data(dest_ip text, ts timestamp, src_ip text);''')
    con.commit()
    return con, cur

con, cur = get_db_connection()

# Use this if program unexpectedly crashes in middle of processing a file
FINISHED_FILES = []

# Add data to sqlite database
for trace_dir in trace_dirs:
    for trace_file in os.listdir(os.path.join(data_dir, trace_dir)):
        if trace_file in FINISHED_FILES:
            print("Skipping {}".format(trace_file))
            sys.stdout.flush()
            continue
        with open(os.path.join(data_dir, trace_dir, trace_file)) as f:
            print("Opening file: {}".format(trace_file))
            header = next(f)
            next(f)
            i = 0
            for line in f:
                start, end, sif, src_ip, src_port, dif, dst_ip, dst_port, protocol, fl, pkts, octets = line.split(',')
                start = datetime.datetime.strptime('2016' + start, '%Y%m%d.%H:%M:%S.%f')
                end = datetime.datetime.strptime('2016' + end, '%Y%m%d.%H:%M:%S.%f')
                command = "INSERT INTO data VALUES ('{}', '{}', '{}')".format(dst_ip, start, src_ip)
                cur.execute(command)
                if i % 100000 == 0:
                    con.commit()
                    print("End of Loop: {}".format(i))
                sys.stdout.flush()
                i += 1
        print("Completed processing for {}".format(trace_file))
        sys.stdout.flush()
    print("Completed Directory for {}".format(trace_dir))
con.close()


# conns = []
# curs = []
# def merge_databases(db1, db2):
#     print("merging")
#     sys.stdout.flush()
#     con3 = sqlite3.connect(db1)

#     con3.execute("ATTACH '" + db2 +  "' as dba")

#     con3.execute("BEGIN")
#     for row in con3.execute("SELECT * FROM dba.sqlite_master WHERE type='table'"):
#         combine = "INSERT OR IGNORE INTO "+ row[1] + " SELECT * FROM dba." + row[1]
#         print(combine)
#         sys.stdout.flush()
#         con3.execute(combine)
#         print("finished execution")
#         sys.stdout.flush()
#     con3.commit()
#     con3.execute("detach database dba")

# merge_databases("data1.db", "data2.db")
# merge_databases("data1.db", "data3.db")

# Write data to compressed npz file
conn = sqlite3.connect("data.db")
cur = conn.cursor()
cur.execute("SELECT DISTINCT dest_ip from data;")
out = cur.fetchall()

processed_data = []
i = 0
for dest_ip in out:
    dest_ip = dest_ip[0]
    command = "SELECT ts, src_ip FROM data WHERE dest_ip='{}' ORDER BY ts;".format(dest_ip)
    cur.execute(command)
    out = cur.fetchall()
    ts = np.asarray([t for (t,e) in out])
    events = np.asarray([e for (t,e) in out])
    processed_data.append(np.column_stack((ts, events)))
    if i % 100 == 0:
        print("FINISHED {} for dest_ip {}".format(i, dest_ip))
        sys.stdout.flush()
np.savez_compressed(output_file, *processed_data)
conn.close()