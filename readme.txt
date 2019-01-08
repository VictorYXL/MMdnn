1 Make sure to run in our custom pycaffe
2 Support crop without axis and offset
3 Support BNMSRA
4 Polish the prototxt manually:
	4.1 Remove unsupport layers until first conv
	4.2 Add input layer and make sure the bottom
	4.3 If you need to merge last 12 inner product layers into 4 conv layers:
		4.3.1 Find stretch layer and modify it into pooling layer
		4.3.2 AVE or MAX are both OK
		4.3.3 kernel_h / kernel_w = The input height / width of stretch layer
		4.3.4 Rename this pooling layer to IP_TO_CONV
		4.3.5 Remove others layer except last 12 InnerProduct layers 
		4.3.6 Make sure InnerProduct layers in order score_g1 delta_g1 landmark_delta_g1 ... score_g4 delta_g4 landmark_delta_g24
	(You can refer Demo.prototxt to modify prototxt)
5 Use v-xianly's mmdnn (https://github.com/VictorYXL/MMdnn/tree/perf/caffe_detection) to convert:
	mmconvert -sf caffe -in xxx.prototxt -iw xxx.caffemodel -df onnx -om xxx.onnx