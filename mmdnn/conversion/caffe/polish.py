import copy
import os
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2


def create_prototxt(model, src_prototxt):
    prototxt_file = open (src_prototxt, 'w')
    model_copied = copy.deepcopy(model)
    for layer in model_copied.layer:
        layer.ClearField('blobs')
    prototxt_file.write(str(model_copied))
    prototxt_file.close()


def rename_top_and_bottom(replace_name_map, layer):
    for bottom_index, name in enumerate(layer.bottom):
        if name in replace_name_map.keys():
            layer.bottom.remove(name)
            layer.bottom.insert(bottom_index, replace_name_map[name])
    for top_index, name in enumerate(layer.top):
        if name in replace_name_map.keys():
            layer.top.remove(name)
            layer.top.insert(top_index, replace_name_map[name])


def add_lost_scale_after_bn(caffe_model):
    src_layers = caffe_model.layer
    dst_layers = caffe_pb2.NetParameter().layer
    replace_name_map = dict()
    # Rename layer input if using bn_output
    for index, layer in enumerate(src_layers):
        rename_top_and_bottom(replace_name_map, layer)
        dst_layers.extend([layer])
        if layer.type == 'BatchNorm' and (index + 1 >= len(src_layers) or src_layers[index + 1].type != "Scale"):
            print('Merge bn + scale in layer ' + str(layer.name))
            # BN (bn_input_name -> scale_input_name) 
            # Scale (scale_input_name -> scale_output_name)
            bn_origin_output_name = layer.top[0]
            scale_input_name = layer.top[0] + '_scale_in'
            scale_output_name = layer.top[0] + '_scale_out'
            # Rename BN output 
            dst_layers[-1].top[0] = scale_input_name
            # update rename dict
            replace_name_map[layer.top[0]] = layer.top[0] + '_scale_out'
            # Create scale layer
            scale_layer = caffe_pb2.LayerParameter()
            scale_layer.name = layer.name + '_scale'
            scale_layer.type = u'Scale'
            scale_layer.bottom.append(scale_input_name)
            scale_layer.top.append(scale_output_name)
            # Add scale and bias blob in scale
            scale_blob = scale_layer.blobs.add()
            scale_blob.shape.dim.append(layer.blobs[0].shape.dim[0])
            for i in range(layer.blobs[0].shape.dim[0]):
                scale_blob.data.append(1)
            bias_blob = scale_layer.blobs.add()
            bias_blob.shape.dim.append(layer.blobs[0].shape.dim[0])
            for i in range(layer.blobs[0].shape.dim[0]):
                bias_blob.data.append(0)
            scale_layer.scale_param.bias_term = True
            # Add scale layer
            dst_layers.extend([scale_layer])
    for index in range(0, len(caffe_model.layer)):
        caffe_model.layer.pop()
    caffe_model.layer.extend(dst_layers)


def split_bnmsra_into_bn_and_scale(caffe_model):
    src_layers = caffe_model.layer
    dst_layers = caffe_pb2.NetParameter().layer
    replace_name_map = dict()
    # Rename layer input if using bn_output
    for layer in src_layers:
        rename_top_and_bottom(replace_name_map, layer)
        if layer.type == 'BatchNormMSRA':
            # BNMSRA 
            # blob[0] -> scale
            # blob[1] -> bias
            # blob[2] -> mean (EX)
            # blob[3] -> E(X^2)
            # var = E(X^2) - (EX)^2 = blob[3] - blob[2] ^ 2
            # BN (bnmsra_input_name -> scale_input_name) 
            # Scale (scale_input_name -> scale_output_name)
            replace_name_map[layer.top[0]] = layer.top[0] + '_scale_out'

            # Create BatchNorm layer
            bn_layer = caffe_pb2.LayerParameter()
            bn_layer.batch_norm_param.eps = 1e-10
            bn_layer.name = layer.name
            bn_layer.type = u'BatchNorm'
            bn_layer.bottom.append(layer.bottom[0])
            bn_layer.top.append(layer.top[0] + '_scale_in')
            mean_blob = bn_layer.blobs.add()
            mean_blob.shape.dim.append(layer.blobs[2].shape.dim[1])
            for i in range(layer.blobs[2].shape.dim[1]):
                mean_blob.data.append(layer.blobs[2].data[i])
            var_blob = bn_layer.blobs.add()
            var_blob.shape.dim.append(layer.blobs[3].shape.dim[1])
            for i in range(layer.blobs[3].shape.dim[1]):
                var_blob.data.append(layer.blobs[3].data[i] - layer.blobs[2].data[i] * layer.blobs[2].data[i])
            moving_coefficient_blob = bn_layer.blobs.add()
            moving_coefficient_blob.shape.dim.append(1)
            moving_coefficient_blob.data.append(1)
            dst_layers.extend([bn_layer])

            # Create scale layer
            scale_layer = caffe_pb2.LayerParameter()
            scale_layer.name = layer.name + '_scale'
            scale_layer.type = u'Scale'
            scale_layer.bottom.append(layer.top[0] + '_scale_in')
            scale_layer.top.append(layer.top[0] + '_scale_out')
            # Add scale and bias blob in scale
            scale_blob = scale_layer.blobs.add()
            scale_blob.shape.dim.append(layer.blobs[0].shape.dim[1])
            for i in range(layer.blobs[0].shape.dim[1]):
                scale_blob.data.append(layer.blobs[0].data[i])
            bias_blob = scale_layer.blobs.add()
            bias_blob.shape.dim.append(layer.blobs[1].shape.dim[1])
            for i in range(layer.blobs[1].shape.dim[1]):
                bias_blob.data.append(layer.blobs[1].data[i])
            scale_layer.scale_param.bias_term = True
            dst_layers.extend([scale_layer])
        else:
            dst_layers.extend([layer])
    for index in range(0, len(caffe_model.layer)):
        caffe_model.layer.pop()
    for index in range(0, len(caffe_model.layer)):
        caffe_model.layer.pop()
    caffe_model.layer.extend(dst_layers)

def check_ip_to_conv(caffemodel):
    src_layers = caffemodel.layer
    if len(src_layers) < 14:
        return False
    elif not (src_layers[-14].type == 'Pooling' and src_layers[-13].type == 'Split' and [layer.type for layer in src_layers[-12:]] == ['InnerProduct'] * 12):
        return False
    elif src_layers[-14].name != 'IP_TO_CONV':
        return False
    else:
        return True

# Must run in custom caffe
def special_polish_ip_to_conv(caffemodel):
    src_layers = caffemodel.layer
    if check_ip_to_conv(caffemodel):
        print('Remove pooling layer: ' + str(src_layers[-14].name))
        for layer in src_layers[-12:]:
            if layer.blobs[0].shape.dim[1] != src_layers[-12].blobs[0].shape.dim[1]:
                raise ValueError('All the inner product has the same channel')
        bottom_name = src_layers[-14].bottom[0]
        top_name = ['score', 'delta', 'landmark_delta']
        # Devide into score delta landmark_delta group
        inner_product_layers = [src_layers[-12::3], src_layers[-11::3], src_layers[-10::3]]
        # Special pre-process channle[1] - channel[0] for score inner product
        for layer in inner_product_layers[0]:
            if layer.blobs[0].shape.dim[0] == 2:
                for i in range(layer.blobs[0].shape.dim[1]):
                    layer.blobs[0].data[i] = layer.blobs[0].data[i + layer.blobs[0].shape.dim[1]] - layer.blobs[0].data[i]
                for i in range(layer.blobs[0].shape.dim[1]):
                    t = layer.blobs[0].data.pop(-1)
                layer.blobs[0].shape.dim[0] = 1
                if len(layer.blobs) > 1:
                     layer.blobs[1].data[0] = layer.blobs[1].data[1] - layer.blobs[1].data[0]
                     t = layer.blobs[1].data.pop(-1)
                     layer.blobs[1].shape.dim[0] = 1
            else:
                raise ValueError('The channel of score inner product Must be 2')
        for i in range(14):
            src_layers.remove(src_layers[-1])
        for i in range(3):
            print('Merge layers ' + str([layer.name for layer in inner_product_layers[i]]) + ' into output_' + top_name[i] + ' layer')
            conv_layer = src_layers.add()
            conv_layer.name = "output_" + top_name[i]
            conv_layer.type = u'Convolution'
            conv_layer.bottom.append(bottom_name)
            conv_layer.top.append('output_' + top_name[i])
            kernel_count = sum(layer.blobs[0].shape.dim[0] for layer in inner_product_layers[i])
            kernel_channel = inner_product_layers[i][0].blobs[0].shape.dim[1]
            conv_layer.convolution_param.num_output = kernel_count
            conv_layer.convolution_param.pad.append(0)
            conv_layer.convolution_param.kernel_size.append(1)
            conv_layer.convolution_param.stride.append(1)
            kernel_blob = conv_layer.blobs.add()
            kernel_blob.shape.dim.append(kernel_count)
            kernel_blob.shape.dim.append(kernel_channel)
            kernel_blob.shape.dim.append(1)
            kernel_blob.shape.dim.append(1)
            for layer in inner_product_layers[i]:
                kernel_blob.data.extend(layer.blobs[0].data)
            if len(layer.blobs) > 1:
                bias_blob = conv_layer.blobs.add()
                bias_blob.shape.dim.append(sum(layer.blobs[1].shape.dim[0] for layer in inner_product_layers[i]))
                for layer in inner_product_layers[i]:
                    bias_blob.data.extend(layer.blobs[1].data)



def caffe_polish(src_model_file, dst_model_file, src_prototxt = None, dst_prototxt = None):
    tmp_model_file = None
    if src_prototxt != None and dst_prototxt != None:
        tmp_model_file = "temp_" + src_model_file
        # Convert caffe_model + prototxt -> temp caffe_model
        net = caffe.Net(src_prototxt, src_model_file, caffe.TEST)
        net.save(tmp_model_file)
        file = open(tmp_model_file, 'rb')
    else:
        file = open(src_model_file, 'rb')

    caffe_model = caffe_pb2.NetParameter()
    caffe_model.ParseFromString(file.read())
    file.close()

    add_lost_scale_after_bn(caffe_model)
    split_bnmsra_into_bn_and_scale(caffe_model)
    special_polish_ip_to_conv(caffe_model)

    file = open(dst_model_file, 'wb')
    file.write(caffe_model.SerializeToString())
    file.close()
    if src_prototxt != None and dst_prototxt != None:
        if tmp_model_file != None and os.path.exists(tmp_model_file):
            os.remove(tmp_model_file)
        create_prototxt(caffe_model, dst_prototxt)
