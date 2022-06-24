import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from model import appnp, sgconv, splineconv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True) ## Cora, PubMed, CiteSeer
parser.add_argument('--split', type=str, default='public') ## public, full, random
parser.add_argument('--model', required=True) ## APPNP, SGConv, SplineConv
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--K_sgconv', type=int, default=2)
parser.add_argument('--K_appnp', type=int, default=50)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

if args.split == 'random':
    dataset = Planetoid(root='/tmp'+args.dataset, name=args.dataset, split=args.split,
                        transform=T.TargetIndegree(), num_train_per_class=170
                        )
else:
    dataset = Planetoid(root='/tmp/'+args.dataset, name=args.dataset, split=args.split,
                        transform=T.TargetIndegree()
                        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)
criterion = torch.nn.CrossEntropyLoss()

app = appnp(dataset=dataset, hidden=args.hidden, K=args.K_appnp, alpha=args.alpha).to(device)
sgc = sgconv(dataset=dataset, K=args.K_sgconv).to(device)
spc = splineconv(dataset=dataset, hidden=args.hidden).to(device)

if args.model == 'APPNP': model = app
elif args.model == 'SGConv': model = sgc
elif args.model == 'SplineConv': model = spc

if args.optimizer == 'Adam': optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'NAdam': optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model.reset_parameters()

def train(model, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(data)[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test(model, optimizer, mask):
    model.eval()
    correct = model(data).argmax(dim=1)[mask] == data.y[mask]
    return int(correct.sum()) / int(mask.sum())

result = []
best_val_acc, best_test_acc = 0, 0

for epoch in range(1, args.epochs+1):
    loss = train(model, optimizer)
    train_acc = test(model, optimizer, data.train_mask)
    val_acc = test(model, optimizer, data.val_mask)
    test_acc = test(model, optimizer, data.test_mask)
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        # print(f'epoch: {epoch} | loss: {loss.item():.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc} | test_acc: {test_acc}')
result.append(best_test_acc)

print(f'dataset: {args.dataset}, split: {args.split}, model: {args.model}, best accuracy: {result[0]}')