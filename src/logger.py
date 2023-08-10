"""
Logger for components of the pipeline
"""
# -*- coding: utf-8 -*-
from datetime import datetime

class Logger:
    """ Logger """
    def __init__(self):
        self.start = None
        self.end = None

    def log_start(self, name):
        """ Start info """
        print(f"Started: {name}")
        self.start = datetime.now()
        print(f"Started at {self.start}")

    def log_end(self):
        """ End info """
        self.end = datetime.now()
        print(f"Ended at {self.end}, took {self.end - self.start}")
