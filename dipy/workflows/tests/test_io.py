from dipy.data import get_data
from dipy.workflows.io import IoInfoFlow


def test_io_info():
    fimg, fbvals, fbvecs=get_data('small_101D')
    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fbvecs])
    
    fimg, fbvals, fvecs = get_data('small_25')
    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fvecs])
    
    
if __name__ == '__main__':
    test_io_info()
