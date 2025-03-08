from utils.format_labels import tensorflow_to_paddleocr
import os

def testConverter():
    bbs_file = "/Users/cvuong/Challenge_Suez/ai meter reading.v2i.tensorflow/train/_annotations.csv"
    labels_file = "/Users/cvuong/Challenge_Suez/index.csv"
    output_file = "/Users/cvuong/Challenge_Suez/paddleocr_annotations.txt"
    img_dir = "/Users/cvuong/Challenge_Suez/pictures"
    tensorflow_to_paddleocr(bbs_file, labels_file, img_dir, output_file)
    assert open(output_file, 'r').read() == open("/Users/cvuong/Challenge_Suez/paddleocr_annotations.txt", 'r').read()
    # os.remove(output_file)
    # assert not os.path.exists(output_file)

if __name__ == "__main__":
    testConverter()
    print("testConverter passed")
