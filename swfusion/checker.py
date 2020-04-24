import logging
import pickle


class Checker(object):
    def __init__(self, CONFIG):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.satel_names = self.CONFIG['satel_names']

        self.check_satel_missing_dates()

    def check_satel_missing_dates(self):
        for name in self.satel_names:
            missing_dates_file = self.CONFIG[name]['files_path'][
                'missing_dates']
            with open(missing_dates_file, 'rb') as f:
                missing_dates = pickle.load(f)

            print((f"""{name}: {missing_dates}"""))
