

class BeuafortScale(object):

    def __init__(self, CONFIG, passwd):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.db_root_passwd = passwd

    def create_table(self):
