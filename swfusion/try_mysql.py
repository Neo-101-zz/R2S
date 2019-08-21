import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///:memory:', echo=True)
Base = declarative_base()
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    fullname = Column(String)
    nickname = Column(String)

    def __repr__(self):
       return "<User(name='%s', fullname='%s', nickname='%s')>" % (
                            self.name, self.fullname, self.nickname)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
Session = sessionmaker()
Session.configure(bind=engine)  # once engine is available
session = Session()

ed_user = User(name = 'ed', fullname='Ed Jones', nickname='edsnickname')
session.add(ed_user)
our_user = session.query(User).filter_by(name='ed').first()

session.add_all([
    User(name='wendy', fullname='Wendy Williams', nickname='windy'),
    User(name='mary', fullname='Mary Contrary', nickname='mary'),
    User(name='fred', fullname='Fred Flintstone', nickname='freddy')])
ed_user.nickname = 'eddie'
session.dirty
session.new
session.commit()

ed_user.name = 'Edwardo'
fake_user = User(name='fakeuser', fullname='Invalid', nickname='12345')
session.add(fake_user)
session.query(User).filter(User.name.in_(['Edwardo', 'fakeuser'])).all()
session.rollback()
session.query(User).filter(User.name.in_(['ed', 'fakeuser'])).all()

config = {
    'user': 'root',
    'password': '39cnj971hw-',
    'host': 'localhost',
    'database': 'test',
    'raise_on_warnings': True,
    'use_pure': True
}

try:
    cnx = mysql.connector.connect(**config)
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
else:
    cnx.close()
