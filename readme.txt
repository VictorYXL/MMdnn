1 Make sure to run in our own pycaffe(in these folder's caffe is based on python3.5)
2 Polish the prototxt manually:
	2.1 Remove ImageNormalization layer
	2.2 Remove crop layer when the input is the multiple of 32 and modify the related layers' bottom
	2.3 Find stretch layer and modify is into pooling layer
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
	2.4 Remove others layer except InnerProduct layer
3 Use v-xianly's mmdnn (https://github.com/VictorYXL/MMdnn/tree/v-xianly/caffe_detection) to convert:
	mmconvert -sf caffe -in xxx.prototxt -iw xxx.caffemodel -df onnx -om xxx.onnx