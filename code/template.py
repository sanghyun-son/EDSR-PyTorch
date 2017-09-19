def setTemplate(args):
    # Set the templates here
    if args.template == 'DIV2K':
        args.trainData = 'DIV2K'
        args.testData = 'DIV2K'
        args.epochs = 300
        args.lrDecay = 200
    
    elif args.template == 'DIV2K_jpeg':
        args.trainData = 'DIV2K_jpeg'
        args.testData = 'DIV2K_jpeg'
        args.epochs = 200
        args.lrDecay = 100

    elif args.template == 'MDSR':
        args.trainData = 'DIV2K'
        args.testData = 'DIV2K'
        args.epochs = 650
        args.lrDecay = 200
