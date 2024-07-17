import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed


import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = (5, 3)

from functools import partial
class MLP(nn.Module):  # nn.Moduleを継承する
    def __init__(self, in_dim, hid_dim, out_dim):  # __init__をoverride
        super(MLP, self).__init__()
        self.linear1 = Dense(in_dim, hid_dim, function=relu)
        self.linear2 = Dense(hid_dim, out_dim, function=softmax)

    def forward(self, x):  # forwardをoverride
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def relu(x):
    x = torch.where(x > 0, x, torch.zeros_like(x))
    return x


def softmax(x):
    x -= torch.cat([x.max(axis=1, keepdim=True).values] * x.size()[1], dim=1)
    x_exp = torch.exp(x)
    return x_exp/torch.cat([x_exp.sum(dim=1, keepdim=True)] * x.size()[1], dim=1)


class Dense(nn.Module):  # nn.Moduleを継承する
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        super().__init__()
        # He Initialization
        # in_dim: 入力の次元数、out_dim: 出力の次元数
        self.W = nn.Parameter(torch.tensor(np.random.uniform(
                        low=-np.sqrt(6/in_dim),
                        high=np.sqrt(6/in_dim),
                        size=(in_dim, out_dim)
                    ).astype('float32')))
        self.b = nn.Parameter(torch.tensor(np.zeros([out_dim]).astype('float32')))
        self.function = function

    def forward(self, x):  # forwardをoverride
        return self.function(torch.matmul(x, self.W) + self.b)


mlp = MLP(2, 3, 2)
in_dim = 784
hid_dim = 200
out_dim = 10
# in_dim = 281
# hid_dim = 3
# out_dim = 10

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
t = torch.tensor([0, 1, 1, 0], dtype=torch.long)



@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)
            t_hot = torch.eye(2)[t]
            y = mlp.forward(x)
            
            # loss = F.cross_entropy(y_pred, y)
            loss = -(t_hot*torch.log(y)).sum(axis=1).mean()
            # train_loss.append(loss.item())
            train_loss.append(loss.tolist())
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # acc = accuracy(y_pred, y)
            pred = y.argmax(1)

            acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
            # acc = accuracy(y_pred, y.argmax(dim=1)) 
            # train_acc.append(acc.item())
            train_acc.append(acc.tolist())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            t_hot = torch.eye(2)[t]
            
            with torch.no_grad():
                y_pred = model(X)
            
            # 順伝播
            y = mlp.forward(x)

            # 誤差の計算(クロスエントロピー誤差関数)
            # loss = F.cross_entropy(y_pred, y)
            loss = -(t_hot*torch.log(y)).sum(axis=1).mean()

            # モデルの出力を予測値のスカラーに変換
            pred = y.argmax(1)

            val_loss.append(loss.tolist())

            # val_loss.append(F.cross_entropy(y_pred, y).item())
            # val_acc.append(accuracy(y_pred, y).tolist())
            val_acc.append(accuracy(y_pred, y).item())
            # val_acc.append(accuracy(y_pred, y.argmax(dim=1)).item()) 

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
