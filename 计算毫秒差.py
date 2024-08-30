import datetime
from datetime import datetime
t1 = datetime.now()
print('121213213213213')
# t2 = "14:27:08.817"
t2 = datetime.now()
deltat_sec = (t2 - t1).seconds
deltat_ms = (t2 - t1).microseconds / 1000
print(deltat_sec)  # t2和t1之间相差0秒
print(deltat_ms)   # t2和t1之间相差960.0毫秒
