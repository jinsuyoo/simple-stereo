import argparse
import os
import random
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import core.datasets as datasets
from core.models import *
from evaluate import validate_things

def parse_config():
    """
    Parse the configuration for the training script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_disp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--model', default='basic',
                        help='select model')
    parser.add_argument('--data_path', default='data/SceneFlowData/',
                        help='data path')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--restore_ckpt', default= None,
                        help='load model')
    parser.add_argument('--save_path', default=None,
                        help='save model')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=310, metavar='S',
                        help='random seed (default: 310)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=6, 
                        help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], 
                        help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.001, 
                        help="max learning rate.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512], 
                        help="size of the random image crops used during training.")
    parser.add_argument('--weight_decay', type=float, default=.00001, 
                        help="Weight decay in optimizer.")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    if args.save_path is None:
        # Generate a save path if not provided based on the current time
        args.save_path = 'checkpoints/' + time.strftime('%Y-%m-%d_%H-%M-%S')

    # Create the save path if it does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"Experiment will be saved in: {args.save_path}")

    return args



def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train():
    """
    Train the model.
    """
    args = parse_config()


    """
    Load the data
    """

    train_loader = datasets.fetch_dataloader(args)

    total_steps = len(train_loader) * args.num_epochs
    print(f'Total number of training steps: {total_steps}')

    """
    Create the model
    """

    if args.model == 'stackhourglass':
        raise NotImplementedError('Stackhourglass model not implemented')
        #model = stackhourglass(args.max_disp)
    elif args.model == 'basic':
        model = basic(args.max_disp)
    else:
        raise ValueError('Invalid model type')

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()
    
    # Print number of model parameters 
    #print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

    # print number of model parameters more elegantly (x.xxM)
    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')


    #if args.loadmodel is not None:
    #    print('Load pretrained model')
    #    pretrain_dict = torch.load(args.loadmodel)
    #    model.load_state_dict(pretrain_dict['state_dict'])

    #print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        total_steps=total_steps+100,
        pct_start=0.01, 
        cycle_momentum=False, 
        anneal_strategy='linear'
    )

    #return

    # Training loop
    for epoch in range(args.num_epochs):
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            valid = valid.bool()

            assert model.training
            depth_predictions = model(image1, image2)
            assert model.training

            # squeeze out the channel dimension (B, 1, H, W) -> (B, H, W)
            depth_predictions = depth_predictions.squeeze(1)
            flow = flow.squeeze(1)

            loss = F.smooth_l1_loss(depth_predictions[valid], flow[valid], reduction='mean')
            
            if torch.isnan(loss) or torch.isinf(loss):
                print('NaN loss encountered. Skipping this batch.')
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Log the loss
            if (i_batch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i_batch}, Loss: {loss.item()}')

            if i_batch == 1000:
                break

        # save the model with filename containing the epoch number 3 digits wide
        current_save_path = os.path.join(args.save_path, f'checkpoint_ep{epoch+1:03d}.pth')
        print(f'Saving model to {current_save_path}')
        torch.save(model.state_dict(), current_save_path)

        """
        validation code
        """
        results = validate_things(model.module)

        #logger.write_dict(results)

        #model.train()
        #model.module.freeze_bn()

    print("FINISHED TRAINING")
    #logger.close()
    #PATH = 'checkpoints/%s.pth' % args.name
    #torch.save(model.state_dict(), PATH)


    """


    start_full_time = time.time()
	for epoch in range(0, args.epochs):
	   print('This is %d-th epoch' %(epoch))
	   total_train_loss = 0
	   adjust_learning_rate(optimizer,epoch)

	   ## training ##
	   for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
	     start_time = time.time()

	     #loss = train(imgL_crop,imgR_crop, disp_crop_L)

         model.train()

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output,1)
            loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()



	     print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
	     total_train_loss += loss
	   print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

	   #SAVE
	   savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
	   torch.save({
		    'epoch': epoch,
		    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
		}, savefilename)

	print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

	#------------- TEST ------------------------------------------------------------
	total_test_loss = 0
	for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
	       test_loss = test(imgL,imgR, disp_L)
	       print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
	       total_test_loss += test_loss

	print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
	#----------------------------------------------------------------------------------
	#SAVE test information
	savefilename = args.savemodel+'testinformation.tar'
	torch.save({
		    'test_loss': total_test_loss/len(TestImgLoader),
		}, savefilename)
    """




if __name__ == '__main__':
   train()
    
