from scripts.img_seg_app import ImageSegApplication
from scripts.img_seg_dataset import test

if __name__ == '__main__':
    app = ImageSegApplication()
    # app.show_sample_data()
    # app.run_training_main_process()
    # app.test_sample()
    app.run_inference()
    # test()
