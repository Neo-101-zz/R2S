from sqlalchemy import *
from sqlalchemy.orm import *
from sqlalchemy.ext.declarative import declarative_base
import random
import itertools

Base = declarative_base()

class Entry(Base):
    __tablename__ = 'a'

    id = Column(Integer, primary_key=True)
    sub = Column(Integer)
    division = Column(Integer)
    created = Column(Integer)
    __table_args__ = (UniqueConstraint('sub', 'division', 'created'), )

# e = create_engine("postgresql://scott:tiger@localhost/test", echo=True)
e = create_engine('sqlite:///:memory:', echo=True)
Base.metadata.drop_all(e)
Base.metadata.create_all(e)


a_bunch_of_fake_unique_entries = list(
    set(
        (random.randint(1, 100000),
         random.randint(1, 100000),
         random.randint(1, 100000)
        )
        for i in range(100000)
    )
)

entries_we_will_start_with = a_bunch_of_fake_unique_entries[0:50000]
entries_we_will_merge = a_bunch_of_fake_unique_entries[30000:100000]

sess = Session(e)

counter = itertools.count(1)
sess.add_all(
    [Entry(id=next(counter), sub=sub, division=division, created=created)
     for sub, division, created in entries_we_will_start_with
    ]
)
sess.commit()

# here's where your example begins... This will also batch it
# to ensure it can scale arbitrarily

while entries_we_will_merge:
    batch = entries_we_will_merge[0:1000]
    entries_we_will_merge = entries_we_will_merge[1000:]
    breakpoint()
    existing_entries = dict(
        (
            (entry.sub, entry.division, entry.created),
            entry
        )
        for entry in sess.query(Entry).filter(
            tuple_(Entry.sub, Entry.division, Entry.created).in_(
                [
                    tuple_(sub, division, created) \
                    for sub, division, created in batch
                ]
            )
        )
    )

    inserts = []
    for entry_to_merge in batch:
        existing_entry = existing_entries.get(entry_to_merge, None)
        if existing_entry:
            # do whatever to update existing
            pass
        else:
            inserts.append(
                dict(
                    id=next(counter),
                    sub=entry_to_merge[0],
                    division=entry_to_merge[1],
                    create_engine=entry_to_merge[2]
                )
            )
    if inserts:
        sess.execute(Entry.__table__.insert(), params=inserts)
