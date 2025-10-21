import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import argparse
import math
import torch
import numpy as np
import random
import time
import torch.nn as nn
import logging
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from shd_dataset import SHD
from models import HetSynRSNNCell, HetSynOutCell
from models.example_nets import HetSynLIFFSNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random number seed")
    parser.add_argument("--thr", type=float, default=1.0, help="threshold")
    parser.add_argument("--p_noise", type=float, default=0.0, help="noise level")
    parser.add_argument("--noise_type", type=str, default='None', choices=['none', 'add', 'del'], help="noise type")
    parser.add_argument("--dropout_rate", nargs='+', type=float, default=[0.2, 0.2], help="dropout_rate")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="max learning rate")
    parser.add_argument("--weight_decay", type=float, default=4e-3, help="weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="learning epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="random number seed")
    parser.add_argument("--plateau_ratio", type=float, default=0.05, help="plateau ratio for cos scheduler")
    parser.add_argument('--data_path', type=str, default='./data/SHD', help='SHD data path')
    parser.add_argument('--no_clip', action='store_true', help='not clip rho')

    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(path_results, dir_name, file_name='train.log', filemode='a', format='%(message)s', level=logging.INFO):
    log_path = Path(path_results, dir_name, file_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(str(log_path))

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode=filemode)
        formatter = logging.Formatter(format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(level)

    return logger


class HetSynLifRSNN(HetSynLIFFSNN):

    def __init__(self, dropout_rate, **kwargs):
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)

    def create_hidden_layer(self):
        self.layer_hidden1 = HetSynRSNNCell(
            self_rec=self.self_rec, bias=self.bias,
            n_in=self.n_in, n_out=self.n_hid[0], learning_w=self.learning_w,
            learning_rho=self.learning_rho,  # @@@ variants: manually forced
            rho_l=self.rho_l, rho_h=self.rho_h, dt=self.dt, thr=self.thr,
            tau_r=self.tau_r, sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng,
            device=self.device, dtype=self.dtype
        )

        self.dp1 = nn.Dropout(p=self.dropout_rate[0])

        self.layer_hidden2 = HetSynRSNNCell(
            self_rec=self.self_rec, bias=self.bias,
            n_in=self.n_hid[0], n_out=self.n_hid[1], learning_w=self.learning_w,
            learning_rho=self.learning_rho,  # @@@ variants: manually forced
            rho_l=self.rho_l, rho_h=self.rho_h, dt=self.dt, thr=self.thr,
            tau_r=self.tau_r, sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng,
            device=self.device, dtype=self.dtype
        )
        self.dp2 = nn.Dropout(self.dropout_rate[1])

    def create_output_layer(self):
        self.layer_out = HetSynOutCell(
            n_in=self.n_hid[1], n_out=self.n_out, learning_w=self.learning_w,
            learning_rho=self.learning_rho,  # @@@ variants: manually forced
            rho_l=self.rho_l, rho_h=self.rho_h, dt=self.dt, thr=self.thr,
            tau_r=self.tau_r, sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng, bias=self.bias,
            device=self.device, dtype=self.dtype
        )

    def align_params(self):
        self.layer_hidden1.align_params()
        self.layer_hidden2.align_params()
        self.layer_out.align_params()

    def create_alias(self):
        self.w_in1 = self.layer_hidden1.w_in
        self.w_rec1 = self.layer_hidden1.w_rec
        self.w_in2 = self.layer_hidden2.w_in
        self.w_rec2 = self.layer_hidden2.w_rec
        self.w_out = self.layer_out.w_in

        self.rho_in1 = self.layer_hidden1.rho_in
        self.rho_rec1 = self.layer_hidden1.rho_rec
        self.rho_in2 = self.layer_hidden2.rho_in
        self.rho_rec2 = self.layer_hidden2.rho_rec
        self.rho_out = self.layer_out.rho_in

        self.b_in1 = self.layer_hidden1.b_in
        self.b_rec1 = self.layer_hidden1.b_rec
        self.b_in2 = self.layer_hidden2.b_in
        self.b_rec2 = self.layer_hidden2.b_rec
        self.b_out = self.layer_out.b_in

    def initialize_params(
            self,
            w_mean=0., w_std=1.0,
            w_mean_rec=0., w_std_rec=1.0,
            tau_mean=1000e-3, tau_std=100e-3,  # @@@ variants: heter init
            rng=None, **kwargs):
        r""" default w init: normal: mean=0.0, std=1.0/sqrt(nin)
        """
        if rng is None:
            rng = self.rng

        self.layer_hidden1.initialize_params(
            w_mean=w_mean, w_std=w_std / (self.n_in ** 0.5),
            w_mean_rec=w_mean_rec, w_std_rec=w_std_rec / (self.n_hid[0] ** 0.5),
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )

        self.layer_hidden2.initialize_params(
            w_mean=w_mean, w_std=w_std / (self.n_hid[0] ** 0.5),
            w_mean_rec=w_mean_rec, w_std_rec=w_std_rec / (self.n_hid[1] ** 0.5),
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )

        self.layer_out.initialize_params(
            w_mean=w_mean, w_std=w_std / (self.n_hid[1] ** 0.5),
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )

    def forward(self, inputs):
        # inputs must be batch_first: b, t, i

        inputs = inputs.transpose(0, 1)  # switch b, t to t, b

        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]

        h1 = None
        h2 = None
        hout = None
        output = 0

        z_outputs = []  # spikes of hidden layer
        mem_outputs = []  # mem potential of the output leaky layer

        for i in range(time_steps):
            xin = inputs[i]
            # h: (hz, hIsyn, hIr)
            h1 = self.layer_hidden1(xin, h1)
            # h2 = self.layer_hidden2(h1[0], h2)
            h2 = self.layer_hidden2(self.dp1(h1[0]), h2)
            z = h2[0]
            z_outputs.append(z)

            vout, hout = self.layer_out(self.dp2(z), hout)
            mem_outputs.append(vout)
            if i > 10:
                output += nn.functional.softmax(vout, dim=1)

        # batch, time_steps, n_out
        y_out = torch.stack(mem_outputs, dim=1)
        z_hidden = torch.stack(z_outputs, dim=1)

        return (y_out, z_hidden, output)


def test():
    model.eval()
    test_acc = 0.
    sum_sample = 0.

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.view((-1)).long().to(device)

            # noise: add or del
            if args.noise_type == 'add' and args.p_noise > 0:
                mask = (torch.rand_like(images) < args.p_noise)
                images[mask] = 1.0
            elif args.noise_type == 'del' and args.p_noise > 0:
                mask = (torch.rand_like(images) < args.p_noise)
                images[mask] = 0.0

            _, _, predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            labels = labels.cpu()
            predicted = predicted.cpu().t()

            test_acc += (predicted == labels).sum()
            sum_sample += predicted.numel()

    return test_acc.data.cpu().numpy() / sum_sample


def train(epochs, criterion, optimizer, scheduler=None):
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    best_model_state = None
    best_acc = 0

    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0
        model.train()
        optimizer.zero_grad()  ######
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'epoch: {epoch + 1}/{epochs}'):
            model.train()
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.view((-1)).long().to(device)

            _, _, predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            train_loss = criterion(predictions, labels) / accumulation_steps
            train_loss.backward()
            train_loss_sum += train_loss.item()

            #########
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

                if not args.no_clip:
                    with torch.no_grad():
                        model.align_params()

            labels = labels.cpu()
            predicted = predicted.cpu().t()

            train_acc += (predicted == labels).sum()
            sum_sample += predicted.numel()

        train_acc = train_acc.data.cpu().numpy() / sum_sample
        valid_acc = test()
        # train_loss_sum += train_loss

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss_sum / len(train_loader))
        test_acc_list.append(valid_acc)


        print('lr: ', optimizer.param_groups[0]["lr"])
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if valid_acc > 0.90:
                save_path = f'./pth/model_{best_acc:.4f}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        msg = ('epoch: {:3d}, Train Loss: {:.2f}, Train Acc: {:.4f}, Valid Acc: {:.4f}, Best Acc: {:.4f}'
               .format(
            epoch,
            train_loss_sum / len(train_loader),
            train_acc,
            valid_acc,
            best_acc
        ))
        print(msg)
        logger.info(msg)
    return train_loss_list, train_acc_list, test_acc_list, best_model_state, best_acc


def get_cos_scheduler(plateau_ratio=0.1):
    plateau_start = int(total_steps * (1 - plateau_ratio))

    def cosine_value(step):
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    plateau_lr_value = cosine_value(plateau_start)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < plateau_start:
            return cosine_value(current_step)
        else:
            return plateau_lr_value

    return lr_lambda

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    set_seed(args.seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    # learning parameters
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epochs = args.epochs
    warmup_ratio = args.warmup_ratio
    data_path = Path(args.data_path)

    train_dir = data_path / 'train_4ms'
    train_files = [str(train_dir / i) for i in os.listdir(train_dir)]

    test_dir = data_path / 'test_4ms'
    test_files = [str(test_dir / i) for i in os.listdir(test_dir)]

    train_dataset = SHD(train_files)
    test_dataset = SHD(test_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    model_args = {
        'self_rec': True,
        'learning_rho': True,
        'bias': False,
        'n_in': 700,
        'n_hid': [128, 64],
        'n_out': 20,
        'dt': 1e-3,
        'learning_w': True,
        'rho_l': 0.0,
        'rho_h': 1.0,
        'thr': args.thr,
        'tau_r': 20e-3,
    }
    model = HetSynLifRSNN(
        dropout_rate=args.dropout_rate,
        rng=rng,
        device=device,
        **model_args,
    )

    with torch.no_grad():
        n_in = model.n_in
        n_hid = model.n_hid
        n_out = model.n_out
        dt = model.dt
        model.initialize_params()

        tau_in1 = torch.empty((n_hid[0], n_in), device=device)
        tau_rec1 = torch.empty((n_hid[0], n_hid[0]), device=device)
        tau_in2 = torch.empty((n_hid[1], n_hid[0]), device=device)
        tau_rec2 = torch.empty((n_hid[1], n_hid[1]), device=device)
        tau_out = torch.empty((n_out, n_hid[1]), device=device)

        tau_in1.uniform_(5e-3, 20e-3, generator=rng)
        tau_rec1.uniform_(5e-3, 20e-3, generator=rng)
        tau_in2.uniform_(5e-3, 20e-3, generator=rng)
        tau_rec2.uniform_(5e-3, 20e-3, generator=rng)
        tau_out.uniform_(5e-3, 20e-3, generator=rng)

        model.rho_in1.data.copy_(torch.exp(-dt / tau_in1))
        model.rho_rec1.data.copy_(torch.exp(-dt / tau_rec1))
        model.rho_in2.data.copy_(torch.exp(-dt / tau_in2))
        model.rho_rec2.data.copy_(torch.exp(-dt / tau_rec2))
        model.rho_out.data.copy_(torch.exp(-dt / tau_out))

        model.align_params()

    model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    base_params = [
        model.w_in1,
        model.w_rec1,
        model.w_in2,
        model.w_rec2,
        model.w_out,
    ]

    rho_params = [
        model.rho_in1,
        model.rho_rec1,
        model.rho_in2,
        model.rho_rec2,
        model.rho_out
    ]

    optimizer = AdamW([
        {'params': base_params, 'lr': learning_rate},
        {'params': rho_params, 'lr': learning_rate * 2},
    ], lr=learning_rate, weight_decay=weight_decay)

    total_steps = 40 * math.ceil(len(train_loader) / accumulation_steps)
    warmup_steps = int(warmup_ratio * total_steps)
    lr_lambda = get_cos_scheduler(args.plateau_ratio)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # logging
    logger = get_logger('./log', 'HetSynLIF_RSNN', f"train_{time.strftime('%Y_%m_%d_%H_%M_%S')}.log")
    logger.info(msg='Training parameters')
    for key, value in vars(args).items():
        logger.info(msg =f"    {key}: {value}")
    logger.info(msg='\nModel parameters')
    for key, value in model_args.items():
        logger.info(f"    {key}: {value}")


    train_loss_list, train_acc_list, test_acc_list, best_model_state, best_acc = train(epochs, criterion, optimizer, scheduler)
    training_logs = {
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,
        'test_acc_list': test_acc_list,
        'model': model.state_dict(),
        'training_parameters': args,
        'model_parameters': model_args,
        'best_model_state': best_model_state,
        'best_acc': best_acc
    }
    save_path = f'./data/HetSynLIF_RSNN/seed_{args.seed}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(training_logs, save_path)

