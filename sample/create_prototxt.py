import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import copy

def create_prototxt(model, prototxt_file_name):
    prototxt_file = open (prototxt_file_name, "w")
    model_copied = copy.deepcopy(model)
    for layer in model_copied.layer:
        layer.ClearField("blobs")
    prototxt_file.write(str(model_copied))

if __name__ == '__main__':
    model = caffe_pb2.NetParameter()
    model.ParseFromString(open("model.caffemodel","rb").read())
    create_prototxt(model, "model.prototxt")