import os

dir = '/Users/lujingze/Programming/SWFusion/data/satel/smap_ncs/'
files = [f for f in os.listdir(dir) if f.endswith('.nc')]

for f in files:
    year = f.split('_')[4]
    month = f.split('_')[5]
    new_dir = f"""{dir}Y{year}/M{month}/"""
    os.makedirs(new_dir, exist_ok=True)
    os.rename(f'{dir}{f}', f'{new_dir}{f}')
