from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import utils

"""
engine = create_engine('mysql+mysqlconnector://root:39cnj971hw-@localhost/SWFusion?use_pure=True', echo=True)
Session = sessionmaker(bind=engine)
session = Session()
DynamicBase = declarative_base(class_registry=dict())
nc_file = '/Users/lujingze/Downloads/NOAA_SFMR20160807I1.nc'
table_name = 'test'
skip_vars = ['DATE', 'TIME']
notnull_vars = ['LAT', 'LON', 'SRR', 'SWS']
unique_vars = []
if engine.has_table(table_name):
    exit(0)
TestTable = utils.create_table_from_netcdf(
    engine, nc_file, table_name, DynamicBase, skip_vars,
    notnull_vars, unique_vars);
test = TestTable()
session.commit()
breakpoint()
"""

from sqlalchemy import Table, MetaData, Column, Integer, String, ForeignKey
from sqlalchemy.orm import mapper

metadata = MetaData()

user = Table('user', metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('fullname', String(50)),
            Column('nickname', String(12))
        )

class User(object):
    def __init__(self, name, fullname, nickname):
        self.name = name
        self.fullname = fullname
        self.nickname = nickname

mapper(User, user)
breakpoint()
