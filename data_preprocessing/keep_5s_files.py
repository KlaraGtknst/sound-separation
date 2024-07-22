import pandas as pd
import os
import zipfile


class PreprocessData:

    def __init__(self, path2annotations:str, path2data:str):
        """
        This class is used to preprocess the data.
        :param path2annotations: path to the annotations
        :param path2data: path to the data
        """

        self.path2annotations = path2annotations
        self.path2data = path2data

    def keep_5s_files(self, max_seconds:int=5):
        annotations = pd.read_csv(self.path2annotations)
        print(f"Number of annotations: {annotations.shape[0]}")
        annotations_5s = annotations[(annotations['End Time (s)'] - annotations['Start Time (s)']) <= max_seconds]
        print(f"Number of annotations after filtering: {annotations_5s.shape[0]}")
        keep_files = annotations_5s['Filename'].values

        # Keep only the files that are in the keep_files list
        if not self.path2data.endswith('zip'):

            for file in os.listdir(self.path2data):
                if file not in keep_files:  # TODO: inefficient
                    #os.remove(os.path.join(self.path2data, file))
                    print(f"Would Removed file: {file}")

        else:
            z = zipfile.ZipFile(self.path2data, "r")
            for filename in z.namelist():
                if filename not in keep_files: # TODO: inefficient
                    #os.remove(os.path.join(self.path2data, file))
                    print(f"Would Removed file: {filename}")