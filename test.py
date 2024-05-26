import time
from torch import nn
from torch.utils.data import DataLoader
from configs import get_args
from data.CIKM.cikm_radar import Radar
from utils import set_seed, make_dir, init_logger

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    from FourCastLSTM import FourCastLSTM

    model = FourCastLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths=args.depths).to(args.device)

    criterion = nn.MSELoss()

    data_root = ''
    test_data = Radar(
        data_type='test',
        data_root=data_root
    )

    test_loader = DataLoader(test_data,
                             num_workers=args.num_workers,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)
    # model.load_state_dict(torch.load('./trained_model_state_dict44'))

    start_time = time.time()

    _, mse, ssim = test(args, logger, 0, model, test_loader, criterion, cache_dir)

    print(f'[Metrics]  MSE:{mse:.4f} SSIM:{ssim:.4f}')
    print(f'Time usage per epoch: {time.time() - start_time:.0f}s')

