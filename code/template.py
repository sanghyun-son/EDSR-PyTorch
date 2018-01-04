def setTemplate(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.lr_decay = 100

    if args.template.find('MDSR') >= 0:
        args.patch_size = 48
        args.epochs = 650

