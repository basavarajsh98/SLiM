import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from src.dataset import prepare_dataset, collate_fn, CustomDataset
from src.model import SLiMedNet
from src.utils import save_model, print_trainable_parameters
from config.config import get_config
from transformers import AutoModelForCausalLM
from src.inference import generate_text
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")

config = get_config()

def train_model(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    step = 0
    for epoch in range(config['epochs']):
        running_loss = 0
        optimizer.zero_grad()

        for i, (input_seq, target_seq, state) in enumerate(dataloader):
            input_seq, target_seq, state = input_seq.to(device), target_seq.to(device), state.to(device)
            
            with autocast(config['device']):
                logits = model(input_seq, state)
                loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
            
            scaler.scale(loss).backward()
            if (i + 1) % config['accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                print(f"Step {step} loss: {(loss.item()):.3f}")
                if step % 50 == 0 or step == config['max_steps']:
                    print("\n=============")
                    print("Baseline: ", generate_text(model, tokenizer, "i feel", max_length=40, state_tensor=None), '\n')
                    print("=============")                    
                    print("Angry German: ", generate_text(model, tokenizer, "i feel", max_length=40, state_tensor=torch.FloatTensor([0,1]).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")
                    print("Angry British: ", generate_text(model, tokenizer, "i feel", max_length=40, state_tensor=torch.FloatTensor([0,0]).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")
                    print("Scared German: ", generate_text(model, tokenizer, "i feel", max_length=40, state_tensor=torch.FloatTensor([1,1]).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")
                    print("Scared British: ", generate_text(model, tokenizer, "i feel", max_length=40, state_tensor=torch.FloatTensor([1,0]).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")
                    print("Loving German: ", generate_text(model, tokenizer, "i feel", max_length=40, state_tensor=torch.FloatTensor([3,1]).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")
                    print("Loving British: ", generate_text(model, tokenizer, "i feel", max_length=40, state_tensor=torch.FloatTensor([3,0]).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============\n")

            running_loss += loss.item()
            if config['max_steps'] and step >= config['max_steps']:
                save_model(model, tokenizer, config['max_steps'], config['save_model_path'])
                return

        epoch_loss = running_loss / len(dataloader)
        epoh_ppx = torch.exp(torch.Tensor([epoch_loss])).item()
        print(f"Epoch {epoch+1} loss: {epoch_loss:.3f} perplexity: {epoh_ppx:.3f}")
        save_model(model, tokenizer, epoch, config['save_model_path'])

if __name__ == "__main__":

    print("Preparing dataset..")
    samples, tokenizer = prepare_dataset()
    dataset = CustomDataset(samples)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

    device = torch.device(config['device'])

    print("\nPreparing model...")
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    for param in model.parameters():
        param.requires_grad = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        fan_in_fan_out=True
    )
    model = get_peft_model(model, peft_config)

    print("\nSLiMing model...")
    model = SLiMedNet(state_embed_dim=config['num_states'], model=model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    print("\nBegin training...\n")
    train_model(model, dataloader, criterion, optimizer, scaler, device)
    print("\nTraining complete!")
