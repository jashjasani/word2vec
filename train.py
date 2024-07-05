from model import SkipGramModel
import torch
import tqdm
from data import create_dataloader_skipgram


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dl = create_dataloader_skipgram("shakepere.txt", window_size=2, num_ns=20, batch_size=128, shuffle=False)
model = SkipGramModel(150, dl.dataset.vocab_size, device=device)
# model.load_state_dict(torch.load("model.pth"))
model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()


num_epochs =  60
for epoch in range(num_epochs):
    total_loss = 0
    for target, context, labels in tqdm.tqdm(dl, desc=f"Training step: {epoch + 1}"):
        target = target.to(device)
        context = context.to(device)
        labels = labels.to(device)
        
        optim.zero_grad()
        dot = model(target, context)
        loss = criterion(dot, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dl)}')

torch.save(model.state_dict(), 'model.pth')