import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': '10m_u_component_of_wind',
        'year': '2019',
        'month': '07',
        'day': '01',
        'time': '00:00',
        'area': [10, 358, 8, 2]
    },
    'download.grib')
