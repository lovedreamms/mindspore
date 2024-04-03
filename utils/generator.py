import mindspore


import network
def create_generator(opt):
    # Initialize the network
    generator = network.KPN(opt.color, opt.burst_length, opt.blind_est, opt.kernel_size, opt.sep_conv, \
        opt.channel_att, opt.spatial_att, opt.upMode, opt.core_bias)
    if opt.load_name == '':
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Load a pre-trained network
        pretrained_net = mindspore.load(opt.load_name)
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    return generator


def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net