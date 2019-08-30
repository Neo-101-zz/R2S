import time
import sys

for i in range(100):
    print(f'\r{100-i}', end='')
    if i == 1:
        sys.stdout.write("\033[K")
    if i == 91:
        sys.stdout.write("\033[K")
    time.sleep(0.05)
# sys.stdout.write('\033[2K\033[1G')
delete_last_lines(1)
print('\rDone')
