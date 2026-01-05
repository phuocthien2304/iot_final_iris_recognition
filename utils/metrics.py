# metrics.py
import torch,torch.nn.functional as F,numpy as np

def evaluate_model(model,loader,device):
    model.eval();losses=[];corr=0;tot=0;logits=[];labels=[]
    ce=torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y,_ in loader:
            x,y=x.to(device),y.to(device)
            o=model(x);l=ce(o,y);losses.append(l.item())
            p=o.argmax(1);corr+=(p==y).sum().item();tot+=y.size(0)
            logits.append(F.softmax(o,1).cpu().numpy());labels.append(y.cpu().numpy())
    return np.mean(losses),corr/tot,np.vstack(logits),np.concatenate(labels),None

def compute_far_frr(logits,labels):
    mp=logits.max(1);pred=logits.argmax(1)
    ths=np.linspace(0,1,101);FARs=[];FRRs=[]
    for t in ths:
        a=mp>=t;g=pred==labels
        FA=np.sum(a&(pred!=labels));FR=np.sum((~a)&g)
        imp=np.sum(pred!=labels);gen=np.sum(pred==labels)
        FAR=FA/imp if imp>0 else 0;FRR=FR/gen if gen>0 else 0
        FARs.append(FAR);FRRs.append(FRR)
    i=np.argmin(np.abs(np.array(FARs)-np.array(FRRs)))
    return (FARs[i]+FRRs[i])/2,ths[i],FARs,FRRs,ths
