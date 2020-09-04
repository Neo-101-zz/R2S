import utils

input_path = ('/Users/lujingze/Programming/SWFusion/data/'
              'dist2coast/dist2coast.txt')
output_path = ('/Users/lujingze/Programming/SWFusion/data/'
               'dist2coast/dist2coast_na_sfmr.txt')

with open(input_path, 'r') as f:
    txt_lines = f.readlines()

north = 50
south = 0
west = 254 - 360
east = 325 - 360
focus_lines = []
total = len(txt_lines)

for idx, line in enumerate(txt_lines):
    print(f'\r{idx+1}/{total}', end='')
    numbers_str = line.split('\t')
    lon = float(numbers_str[0])
    lat = float(numbers_str[1])

    if lon < west or lon > east:
        continue
    if lat < south or lat > north:
        continue
    focus_lines.append(line)

with open(output_path, 'w') as f:
    f.writelines(focus_lines)

utils.delete_last_lines()
print('Done')
