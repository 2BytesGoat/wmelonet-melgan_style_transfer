import yaml
import time
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mel2wav.dataset import AudioDataset
from mel2wav.modules import Audio2Mel, Generator
from mel2wav.utils import save_sample

from travelgan.trainer import TravelGAN

from utils import get_device, load_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--load_path", default=None)

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--n_mel_channels", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    hparams = load_json('./configs', args.config_path)

    # Get CUDA/CPU device
    device = get_device(0)

    root = Path(args.save_path)
    root.mkdir(parents=True, exist_ok=True)
    load_root = Path(args.load_path) if args.load_path else None

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    #######################
    # Load PyTorch Models #
    #######################
    model = TravelGAN(hparams['model'], device=device)
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).cuda()

    # Load MelGAN model
    netG = Generator(args.n_mel_channels, ngf=32, n_residual_layers=3).to(device)
    netG.load_state_dict(torch.load(load_root / "netG.pt"))

    print('Loading data...')
    #############################
    # Create train data loaders #
    #############################
    sequence_length = args.seq_len * 4

    source_train = AudioDataset(
        Path(args.data_path) / "source_train_files.txt", sequence_length, sampling_rate=22050
    )
    style_train = AudioDataset(
        Path(args.data_path) / "style_train_files.txt", sequence_length, sampling_rate=22050
    )

    so_tr_loader = DataLoader(source_train, batch_size=args.batch_size, num_workers=4)
    st_tr_loader = DataLoader(style_train, batch_size=args.batch_size, num_workers=4)

    #############################
    # Create test data loaders #
    #############################

    source_test = AudioDataset(
        Path(args.data_path) / "source_test_files.txt",
        sequence_length,
        sampling_rate=22050,
        augment=False,
    )
    style_test = AudioDataset(
        Path(args.data_path) / "style_test_files.txt",
        sequence_length,
        sampling_rate=22050,
        augment=False,
    )

    so_te_loader = DataLoader(source_test, batch_size=1)
    st_te_loader = DataLoader(style_test, batch_size=1)

    print('Exporting test examples...')
    ##########################
    # Dumping original audio #
    ##########################
    source_voc = []
    style_voc = []
    for i, [so_audio, st_audio] in enumerate(zip(so_te_loader, st_te_loader)):
        # Loading on device
        so_audio = so_audio.cuda()
        st_audio = st_audio.cuda()

        # Transforming to spectograms
        so_mel = fft(so_audio).detach()
        st_mel = fft(st_audio).detach()

        source_voc.append(so_mel.cuda())
        style_voc.append(st_mel.cuda())

        with torch.no_grad():
            soruce_audio = netG(so_mel).squeeze().cpu()
            style_audio = netG(st_mel).squeeze().cpu()
        save_sample(root / ("source_%d.wav" % i), 22050, soruce_audio)
        save_sample(root / ("style_%d.wav" % i), 22050, style_audio)
        # writer.add_audio("source/sample_%d.wav" % i, soruce_audio, 0, sample_rate=22050)
        # writer.add_audio("style/sample_%d.wav" % i, style_audio, 0, sample_rate=22050)

        if i == args.n_test_samples - 1:
            break

    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    print('Commencing training procedure...')
    steps = 0
    for epoch in range(1, args.epochs + 1):
        # Run one epoch
        dis_losses, gen_losses = [], []
        for iterno, [so_t, st_t] in enumerate(zip(so_tr_loader, st_tr_loader)):
            # Loading on device
            so_t = so_t.cuda()
            st_t = st_t.cuda()

            # Transforming to spectograms
            with torch.no_grad():
                so_t = fft(so_t).detach().unsqueeze(1)
                st_t = fft(st_t).detach().unsqueeze(1)

            # Calculate losses and update weights
            dis_loss = model.dis_update(so_t, st_t)
            gen_loss = model.gen_update(st_t, so_t)
            dis_losses.append(dis_loss)
            gen_losses.append(gen_loss)

            ######################
            # Update tensorboard #
            ######################

            steps += 1

            if steps % args.save_interval == 0:
                st = time.time()
                dst_path = root / f'steps_{steps}'
                dst_path.mkdir(exist_ok=True)

                with torch.no_grad():
                    for i, [so_t, st_t] in enumerate(zip(so_te_loader, st_te_loader)):
                        # Loading on device
                        so_t = so_t.cuda()
                        st_t = st_t.cuda()
                        
                        # Transforming to spectograms
                        so_t = fft(so_t).detach().unsqueeze(1)
                        st_t = fft(st_t).detach().unsqueeze(1)

                        # We obtain source to style and style to source
                        so_to_st, st_to_so = model(so_t, st_t)
                        pred_stylized = netG(so_to_st.squeeze(1)).squeeze().cpu()
                        pred_sourced  = netG(st_to_so.squeeze(1)).squeeze().cpu()

                        save_sample(dst_path / ("stylized_%d.wav" % i), 22050, pred_stylized)
                        save_sample(dst_path / ("sourced_%d.wav" % i), 22050, pred_sourced)

                        writer.add_audio(
                            "generated/stylized_%d.wav" % i,
                            pred_stylized,
                            epoch,
                            sample_rate=22050,
                        )
                        writer.add_audio(
                            "generated/sourced_%d.wav" % i,
                            pred_sourced,
                            epoch,
                            sample_rate=22050,
                        )

                model.save(dst_path, epoch)

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} | ms/batch {:5.2f} | dis_loss {} | gen_loss {}".format(
                        epoch,
                        iterno,
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(dis_losses).mean(0),
                        np.asarray(gen_losses).mean(0),
                    )
                )
                start = time.time()


if __name__ == "__main__":
    main()
