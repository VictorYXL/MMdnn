import onnx
def convert_to_variable_length_input_for_pva_net(onnx_model):
    print('Set input and output height and width to \'* ')
    for input in onnx_model.graph.input:
        input.type.tensor_type.shape.dim[1].dim_param = '*'
        input.type.tensor_type.shape.dim[2].dim_param = '*'
    for output in onnx_model.graph.output:
        output.type.tensor_type.shape.dim[1].dim_param = '*'
        output.type.tensor_type.shape.dim[2].dim_param = '*'


def onnx_polish(onnx_model):
    convert_to_variable_length_input_for_pva_net(onnx_model)