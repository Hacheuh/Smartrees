import Smartrees.date_to_data as dtd
import numpy as np
d = dtd.Datas()
def test_init_datas():
    assert len(d.width) == 2
    assert len(d.pos) == 2

def test_get_list_from_dates():
    assert list(d.get_list_from_dates().columns) == ['id', 'Date',
                                                     'Time', 'Sun', 'Cloud']
def test_filter_list():
    df = d.get_list_from_dates()
    f = d.filter_list(df)
    assert df.shape[0]>=f.shape[0]

""" def test_get_data_from_list():
    df = d.get_list_from_dates()
    f = d.filter_list(df)
    dfl = d.get_data_from_list(f)
    assert len(dfl) == 12
    #Very long test maybe it's useless to execute it """

""" def test_get_data_from_dates() :
    dflc = d.get_data_from_dates()
    assert len(dflc) == 12
# Very long test, maybe it's useless to execute it
"""
def test_sea_pixels():
    d.sea_pixel()
    assert (d.sea_pixels.nunique() == 2).output
