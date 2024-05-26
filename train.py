from torch import nn

from data.CIKM.cikm_radar import Radar
from utils import *
from configs import get_args
from torch.utils.data import DataLoader


def setup(args):
    from FourCastLSTM import FourCastLSTM
    model = FourCastLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths=args.depths).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    data_root = ''
    train_data = Radar(
        data_type='train',
        data_root=data_root,
    )
    valid_data = Radar(
        data_type='validation',
        data_root=data_root
    )
    train_loader = DataLoader(train_data,
                              num_workers=args.num_workers,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              drop_last=False,
                              pin_memory=True)
    valid_loader = DataLoader(valid_data,
                              num_workers=args.num_workers,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True)

    return model, criterion, optimizer, train_loader, valid_loader


def main():
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    model, criterion, optimizer, train_loader, valid_loader = setup(args)

    train_losses, valid_losses = [], []

    best_metric = (0, float('inf'), float('inf'))

    for epoch in range(args.epochs):

        start_time = time.time()
        train_loss = train(args, logger, epoch, model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        plot_loss(train_losses, 'train', epoch, args.res_dir, 1)

        if (epoch + 1) % args.epoch_valid == 0:

            valid_loss, mse, ssim = test(args, logger, epoch, model, valid_loader, criterion, cache_dir)

            valid_losses.append(valid_loss)

            plot_loss(valid_losses, 'valid', epoch, args.res_dir, args.epoch_valid)


            if mse < best_metric[1]:
                torch.save(model.state_dict(), f'{model_dir}/trained_model_state_dict')
                best_metric = (epoch, mse, ssim)
            else:
                logger.info(f'[Current Epoch] EP:{epoch:04d} MSE:{mse:.4f} SSIM:{ssim:.4f}')
                if mse < 300:
                    print('save...')
            logger.info(f'[Current Best] EP:{best_metric[0]:04d} MSE:{best_metric[1]:.4f} SSIM:{best_metric[2]:.4f}')

        print(f'Time usage per epoch: {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    main()
