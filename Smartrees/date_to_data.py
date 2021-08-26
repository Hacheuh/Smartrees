from Smartrees.get_dataFrame import SmarTrees
from Smartrees.ee_query import get_meta_data, cloud_out, mapper
""" This class is used to get the global dataframes from images of pos (base is Nice)
between the date_start and date_stop. The images are chosen at a base scale of 30
and only images with a cloud coverage inferior to perc (base 20) are kept
Unique_days is default to 1 and means you don't keep more than one image for each day
"""
""" minimum code:
data_getter=Datas()
dict_of_df=data_getter.get_data_from_dates()
"""


class Datas():
    """Used to generate a dictionnary of  dataframes referenced by the ee_image name's as the key
    It contains Temperature and NDVI"""
    def __init__(self,
                 date_start='2020-07-31',
                 date_stop='2021-01-31',
                 pos=[7.28045496, 43.70684086],
                 perc=20,
                 scale=30,
                 Unique_days=1):
        self.date_start = date_start
        self.date_stop = date_stop
        self.perc = perc
        self.pos = pos
        self.scale = 30
        self.Unique_days = Unique_days

    def get_list_from_dates(self):
        """ gets a dataframe of ee_Images of the position pos taken between date_start and date_stop """
        df_image_list = get_meta_data(self.date_start, self.date_stop,
                                      self.pos)
        return df_image_list.sort_values('Date')

    def filter_list(self, df):
        """ Filtering the images in the list of images based on various arguments """

        # cloud coverage
        df_output = cloud_out(df, perc=20)

        # Keep less cloudy image of each day if Unique_days==1
        if self.Unique_days == 1:

            df_output_list = []
            for date in df_output['Date']:
                df_output_list.append(date)
            df_intermediary = df_output.copy()

            for date in df_output_list:

                if len(df_intermediary[df_intermediary['Date'] == date]) > 1:
                    print(date)
                    indexes = df_intermediary[df_intermediary['Date'] ==
                                              date].index
                    best_index = indexes[0]
                    best_coverage = df_intermediary.loc[best_index]['Cloud']
                    for index in indexes:
                        if df_intermediary.loc[index]['Cloud'] < best_coverage:
                            best_index = index
                            best_coverage = df_intermediary.loc[best_index][
                                'Cloud']

                    for index in indexes:
                        if index != best_index:
                            df_intermediary.drop(index, inplace=True)
            df_output = df_intermediary

        return df_output

    def get_data_from_list(self, df):
        """ Long function that outputs a dictionnary of dataframes containing NDVI and Temperature """
        print(
            f" the dataframe contains {df.shape[0]} lines. considering a mean treatment of 4s, it would take approximately {4.5*df.shape[0]/60} minutes"
        )
        output = {}
        i = 0
        for name in df['id']:
            if i % 10 == 0:
                print(f"file {i} / {df['id'].shape[0]}")
            data = SmarTrees(name, scale=self.scale)
            output[name] = data.get_NDVIandKELVIN()
            i += 1

        return output

    def get_data_from_dates(self):
        df = self.get_list_from_dates()
        df = self.filter_list(df)
        dict_df = self.get_data_from_list(df)
        return dict_df
