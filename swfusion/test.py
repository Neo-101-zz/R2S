
    def _extract_bytemap_to_table_2(self, satel_name, bm_file, table_class,
                                    missing):
        dataset = utils.dataset_of_daily_satel(satel_name, bm_file)
        vars = dataset.variables

        min_lat, max_lat = self.region[0], self.region[1]
        min_lon, max_lon = self.region[2], self.region[3]
        min_lat_idx, max_lat_idx = utils.find_index([min_lat, max_lat], 'lat')
        lat_indices = [x for x in range(min_lat_idx, max_lat_idx+1)]
        min_lon_idx, max_lon_idx = utils.find_index([min_lon, max_lon], 'lon')
        lon_indices = [x for x in range(min_lon_idx, max_lon_idx+1)]

        lat_len = len(lat_indices)
        lon_len = len(lon_indices)
        total = 2 * lat_len * lon_len
        count = 0

        # Store all rows
        whole_table = []

        st = time.time()
        iasc = [0, 1]
        # iasc = 0 (morning, descending passes)
        # iasc = 1 (evening, ascending passes)
        for i in iasc:
            for j in lat_indices:
                for k in lon_indices:
                    count += 1
                    # if count % 2000 == 0:
                    #     progress = float(count)/total*100
                    #     print('\r{:.1f}%'.format(progress), end='')
                    # if not valid_func(vars, i, j, k):
                    if vars['nodata'][i][j][k]:
                        continue
                    table_row = table_class()
                    lat = vars['latitude'][j]
                    lon = vars['longitude'][k]
                    if (not lat or not lon
                        or lat == missing or lon == missing
                        or lat < min_lat or lat > max_lat
                        or lon < min_lon or lon > max_lon):
                        continue
                    # setattr(table_row, lat_name, float(lat))
                    # setattr(table_row, lon_name, float(lon))
                    table_row.latitude = float(lat)
                    table_row.longitude = float(lon)
                    # Set datetime
                    try:
                        mingmt = float(vars['mingmt'][i][j][k])
                        # See note about same mingmt for detail
                        if (mingmt == missing
                            or vars['mingmt'][0][j][k] == \
                            vars['mingmt'][1][j][k]):
                            continue
                        time_str ='{:02d}{:02d}00'.format(
                            *divmod(int(mingmt), 60))
                        bm_file_name = bm_file.split('/')[-1]
                        date_str = bm_file_name.split('_')[1][:8]

                        if time_str.startswith('24'):
                            date_ = datetime.datetime.strptime(
                                date_str + '000000', '%Y%m%d%H%M%S').date()
                            time_ = datetime.time(0, 0, 0)
                            date_ = date_ + datetime.timedelta(days=1)
                            datetime_ = datetime.datetime.combine(
                                date_, time_)
                        else:
                            datetime_ = datetime.datetime.strptime(
                                date_str + time_str, '%Y%m%d%H%M%S')
                    except Exception as msg:
                        breakpoint()
                        exit(msg)
                    # Period check
                    if not datetime_ or not utils.check_period(datetime_,
                                                         self.period):
                        continue
                    table_row.datetime = datetime_

                    table_row.space_time = '%s %f %f' % (datetime_, lat, lon)

                    valid = True
                    table_row.land = bool(vars['land'][i][j][k])
                    table_row.ice = bool(vars['ice'][i][j][k])
                    if satel_name == 'ascat' or satel_name == 'qscat':
                        table_row.windspd = float(vars['windspd'][i][j][k])
                        table_row.winddir = float(vars['winddir'][i][j][k])
                        if (table_row.windspd is None 
                            or table_row.winddir is None
                            or table_row.windspd == missing
                            or table_row.winddir == missing):
                            continue
                        table_row.scatflag = float(vars['scatflag'][i][j][k])
                        table_row.radrain = float(vars['radrain'][i][j][k])
                        if satel_name == 'ascat':
                            table_row.sos = float(vars['sos'][i][j][k])
                    elif satel_name == 'wsat':
                        table_row.w_aw = float(vars['w-aw'][i][j][k])
                        table_row.wdir = float(vars['wdir'][i][j][k])
                        if (table_row.w_aw is None 
                            or table_row.wdir is None
                            or table_row.w_aw == missing
                            or table_row.wdir == missing):
                            continue
                        table_row.vapor = float(vars['vapor'][i][j][k])
                        table_row.cloud = float(vars['cloud'][i][j][k])
                        table_row.rain = float(vars['rain'][i][j][k])
                        table_row.w_lf = float(vars['w-lf'][i][j][k])
                        table_row.w_mf = float(vars['w-mf'][i][j][k])
                    else:
                        sys.exit('satel_name is wrong.')

                    if valid:
                        whole_table.append(table_row)

        return whole_table
