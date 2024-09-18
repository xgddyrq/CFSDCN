def set_template(args):
    if args.template.find('cfsdcn') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.max_epoch = 1000