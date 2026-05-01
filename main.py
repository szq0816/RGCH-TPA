from utils import get_args
from train_asym import TrainerAsym


def main(arg):
    TrainerAsym(arg)


if __name__ == "__main__":

    for dataset in ['mirflickr', 'coco', 'nuswide']:
        args = get_args()
        args.dataset = dataset
        if args.dataset == 'mirflickr':
            args.lr = 0.001
            args.mu = 15
            args.alpha = 5
            args.gamma1 = 1
            args.gamma2 = 5
            args.valid_freq = 3
            args.epochs = 30
            args.hyper_recon = 0.001
            args.query_num = 2000
            args.train_num = 10000
            args.caption_file = "mat/caption.mat"
        elif args.dataset == 'nuswide':
            args.lr = 0.002
            args.mu = 10
            args.alpha = 10
            args.gamma1 = 1
            args.gamma2 = 3
            args.valid_freq = 3
            args.epochs = 30
            args.hyper_recon = 0.001
            args.query_num = 2100
            args.train_num = 10500
            args.caption_file = "mat/caption.txt"
        elif args.dataset == 'coco':
            args.lr = 0.002
            args.mu = 15
            args.alpha = 5
            args.gamma1 = 5
            args.gamma2 = 5
            args.valid_freq = 3
            args.epochs = 30
            args.hyper_recon = 0.005
            args.query_num = 5000
            args.train_num = 10000
            args.caption_file = "mat/caption.mat"

        full = [0.1, 0.3, 0.5]
        oimg = [0.45, 0.35, 0.25]

        import datetime

        _time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not args.is_train:
            _time += "_test"

        for i in [0, 1, 2]:
            args.full_ratio = full[i]
            args.oimg_ratio = oimg[i]

            args.save_dir = f"./result/{args.dataset}/(RGCH-TPA){_time}/full-{args.full_ratio}"

            main(args)

