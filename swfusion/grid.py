import logging
import os
import pickle

from sqlalchemy import create_engine, extract
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, DateTime, Boolean
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import mapper
from global_land_mask import globe

import utils

Base = declarative_base()

class Grid(Base):
    __tablename__ = 'grid'

    key = Column(Integer(), primary_key=True)
    x = Column(Integer(), nullable=False)
    y = Column(Integer(), nullable=False)
    lon = Column(Float(), nullable=False)
    lat = Column(Float(), nullable=False)
    land = Column(Boolean(), nullable=False)
    x_y = Column(String(30), nullable=False, unique=True)
    lon1 = Column(Float(), nullable=False)
    lon2 = Column(Float(), nullable=False)
    lat1 = Column(Float(), nullable=False)
    lat2 = Column(Float(), nullable=False)

class GridManager(object):

    def __init__(self, CONFIG, region, passwd, run):
        self.logger = logging.getLogger(__name__)

        self.CONFIG = CONFIG
        self.region = region
        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        self.spa_resolu = self.CONFIG['grid']['spatial_resolution']

        utils.setup_database(self, Base)

        if run:
            self.setup_grid()

    def setup_grid(self):
        # Create grid table
        # Grid = self.create_grid_table()
        Base.metadata.create_all(self.engine)

        lons, lats = self.gen_lons_lats()

        xs = [x for x in range(len(lons))]
        ys = [y for y in range(len(lats))]

        save_pickle = [
            {'name': 'lons', 'var': lons},
            {'name': 'lats', 'var': lats},
            {'name': 'x', 'var': xs},
            {'name': 'y', 'var': ys}
        ]

        for name_var_pair in save_pickle:
            name = name_var_pair['name']
            var = name_var_pair['var']

            pickle_path = self.CONFIG['grid']['pickle'][name]
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(var, f)

        total = len(lons)
        half_edge = 0.5 * self.spa_resolu

        grid_pts = []
        self.logger.info(f'Generating grid')
        # Traverse lon
        for lon_idx, lon in enumerate(lons):
            print(f'\r{lon_idx+1}/{total}', end='')
            # Traverse lat
            for lat_idx, lat in enumerate(lats):
                # Cal y and x
                pt = Grid()
                pt.x = lon_idx
                pt.y = lat_idx
                pt.x_y = f'{pt.x}_{pt.y}'
                pt.lon = lon
                pt.lat = lat
                pt.lon1, pt.lon2 = pt.lon - half_edge, pt.lon + half_edge
                pt.lat1, pt.lat2 = pt.lat - half_edge, pt.lat + half_edge
                # Check whether the point is ocean or not
                pt.land = bool(globe.is_land(lat, lon))

                grid_pts.append(pt)

        utils.delete_last_lines()
        print('Done')
        # Bulk insert
        utils.bulk_insert_avoid_duplicate_unique(
            grid_pts, self.CONFIG['database']\
            ['batch_size']['insert'],
            Grid, ['x_y'], self.session,
            check_self=True)

    def create_grid_table(self):
        table_name = f'grid'

        class Grid(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Grid, t)

            return Grid

        cols = []
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('x', Integer, nullable=False))
        cols.append(Column('y', Integer, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))
        cols.append(Column('land', Boolean, nullable=False))
        cols.append(Column('x_y', String(30), nullable=False,
                    unique=True))
        cols.append(Column('lon1', Float, nullable=False))
        cols.append(Column('lon2', Float, nullable=False))
        cols.append(Column('lat1', Float, nullable=False))
        cols.append(Column('lat2', Float, nullable=False))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(Grid, t)

        self.session.commit()

        return Grid

    def gen_lons_lats(self):
        decimal_num = str(self.spa_resolu)[::-1].find('.')

        lons = [round(y * self.spa_resolu + self.lon1, decimal_num)
                for y in range(int(
                    (self.lon2 - self.lon1) / self.spa_resolu) + 1)]

        lats = [round(x * self.spa_resolu + self.lat1, decimal_num)
                for x in range(int((
                    self.lat2 - self.lat1) / self.spa_resolu) + 1)]

        return lons, lats
