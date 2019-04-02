1 Make sure to run in our custom pycaffe
2 Support crop without axis and offset
3 Support BNMSRA
4 Create the prototxt by create_prototxt.py 
(modify the caffe model name and output prototxt name when needed in create_prototxt.py)
5 Polish the prototxt manually:
	5.1 Remove unsupport layers until first conv
	5.2 Add input layer and make sure the bottom
	5.3 If you need to merge last 12 inner product layers into 4 conv layers:
		5.3.1 Find last split layer before stretch layer and modify it into pooling layer
		5.3.2 AVE or MAX are both OK
		5.3.3 kernel_h / kernel_w = The input height / width of stretch layer
		5.3.4 Rename this pooling layer to IP_TO_CONV
		5.3.5 Remove others layer except last 12 InnerProduct layers 
		5.3.6 Make sure InnerProduct layers in order score_g1 delta_g1 landmark_delta_g1 ... score_g4 delta_g4 landmark_delta_g24
        (You can use to_copy.txt file to cover the terget prototxt file)
5 use convertor to convert:
	python detection_model_convertor.py -sf caffe -in xxx.prototxt -iw xxx.caffemodel -df onnx -om xxx.onnx
    or python detection_model_convertor.py (the default model will be model.caffemodel , prototxt will be model.prototxt and output will be model.onnx)