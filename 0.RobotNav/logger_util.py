import csv
import time

class CSVLogger:
    def __init__(self, filename, headers):
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(headers)
    def log(self, *row):
        self.writer.writerow(row)
    def close(self):
        self.file.close()