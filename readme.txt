1 Make sure to run in our custom pycaffe
2 Polish the prototxt manually:
	2.1 Remove ImageNormalization layer
	2.2 Support crop without axis and offset
	2.3 Find stretch layer and modify it into pooling layer
		AVE or MAX are both OK
		kernel_h = The input height of stretch layer
		kernel_w = The input width of stretch layer
		stride = 1
		Fot example:
		When blob is 3*56*56 before stretch layer, modify like this.
		layer {
			name: "stretched_rpn_head_3x3"
			type: "Stretch"
			bottom: "rpn_head_3x3"
			top: "stretched_rpn_head_3x3"
		}
		layer {
			name: "stretched_rpn_head_3x3"
			type: "Pooling"
			bottom: "rpn_head_3x3"
			top: "stretched_rpn_head_3x3"
			pooling_param {
				pool: AVE
				kernel_h: 56
				kernel_w: 56
				stride: 1
			}
		}
	2.4 Remove others layer except last 12 InnerProduct layers 
	2.5 Make sure InnerProduct layers in order score_g1 delta_g1 landmark_delta_g1 ... score_g4 delta_g4 landmark_delta_g24
	(You can modify the Demo.prototxt and cover the rest layers by it)
3 Use v-xianly's mmdnn (https://github.com/VictorYXL/MMdnn/tree/perf/caffe_detection) to convert:
	mmconvert -sf caffe -in xxx.prototxt -iw xxx.caffemodel -df onnx -om xxx.onnx