import torch
import torch.optim as optim
import torch.nn as nn

def local_train(model,data,labels,epoch=5,lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=lr)
    model.train()
    
    for _ in range(epoch):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
    return model.state_dict()


def evaluate_model(model,data,labels):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        mse = nn.MSELoss()(outputs,labels)
    return mse.item()



def average_models(states):
    new_state = {}
    for key in states[0].keys():
        new_state[key] = sum(state[key] for state in states) /len(states)
        
    return new_state



        
    
