import argparse
from pathlib import Path
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    parser=argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 on a labeled image dataset.")
    parser.add_argument("--data", type=Path, required=True, help="Root containing train/ and val/, each with fake/ and real/ folders")
    parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("models/best.pt")); args=parser.parse_args()
    train_tf=transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(.12,.12,.12,.04), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train=datasets.ImageFolder(args.data/"train", transform=train_tf); valid=datasets.ImageFolder(args.data/"val", transform=eval_tf)
    if train.classes != ["fake", "real"] or valid.classes != train.classes: raise ValueError("Expected fake/ and real/ classes in both splits.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0); valid_loader=DataLoader(valid,batch_size=args.batch_size,num_workers=0)
    model=timm.create_model("tf_efficientnet_b0",pretrained=True,num_classes=2).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4); loss_fn=nn.CrossEntropyLoss(); best=0.0
    for epoch in range(args.epochs):
        model.train(); total=correct=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device); optimizer.zero_grad(); logits=model(images); loss=loss_fn(logits,labels); loss.backward(); optimizer.step()
            total+=len(labels); correct+=(logits.argmax(1)==labels).sum().item()
        model.eval(); seen=hits=0
        with torch.inference_mode():
            for images,labels in valid_loader:
                logits=model(images.to(device)); labels=labels.to(device); seen+=len(labels); hits+=(logits.argmax(1)==labels).sum().item()
        score=hits/max(seen,1); print(f"epoch {epoch+1}/{args.epochs} train_accuracy={correct/max(total,1):.3f} validation_accuracy={score:.3f}")
        if score>best:
            best=score; args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({"architecture":"tf_efficientnet_b0","classes":train.classes,"state_dict":model.cpu().state_dict()},args.output); model.to(device)
    print(f"Best validation accuracy: {best:.3f}; checkpoint: {args.output}")
if __name__ == "__main__": main()
