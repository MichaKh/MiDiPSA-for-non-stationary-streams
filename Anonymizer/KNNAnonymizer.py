import subprocess
import os
from Anonymizer.AAnonymizer import AAnonymizer


class KNNAnonymizer(AAnonymizer):
    MOA_PATH = 'C:\Users\micha\Desktop\moa-release-2013.11\moa-release-2013.11'

    def __init__(self, stream_path, data_name, evaluation_path, output_path, k, e, b):
        self.stream_path = stream_path
        self.data_name = data_name
        self.output_path = output_path
        self.evaluation_path = evaluation_path
        self.k_anonymity = k
        self.eps = e
        self.buffer_size = b

    def anonymize(self):

        command = ['java', '-cp',
                   'moa.jar;moa-ppsm-0.0.1-SNAPSHOT.jar',
                   'moa.DoTask',
                   'Anonymize',
                   '-s',
                   '(ArffFileStream ',
                   '-f ',
                   str(self.stream_path) + ')',
                   '-f',
                   '(differentialprivacy.DifferentialPrivacyFilter ',
                   '-k',
                   str(self.k_anonymity),
                   '-e',
                   str(self.eps),
                   '-b',
                   str(self.buffer_size) + ')',
                   '-r',
                   self.evaluation_path,
                   '-e',
                   self.evaluation_path + "_full",
                   '-z',
                   '-a',
                   self.output_path]
        subprocess.call(command, cwd=self.MOA_PATH, shell=True)
