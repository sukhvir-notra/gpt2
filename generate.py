import torch
import tiktoken
from gpt2_train_cuda import *

#detect device
torch.manual_seed(1337)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = "mps"
    torch.mps.manual_seed(1337)

print(f"[+] Using device = {device}")


# Load the model
model = torch.load('log/model_19072.pt', map_location=device)
model.eval()  # Set the model to evaluation mode

# Load the GPT-2 tokenizer
enc = tiktoken.get_encoding('gpt2')

# Define the input text
input_text = "It is going to be winter soon"
input_ids = enc.encode(input_text)

# Convert to tensor
input_ids_tensor = torch.tensor([input_ids])

# Generate text completions
with torch.no_grad():
    output = model.generate(
        input_ids_tensor, 
        max_length=len(input_ids) + 50,  # Generate 50 more tokens
        num_return_sequences=5,          # Generate 5 examples
        do_sample=True,                  # Sampling for variety
        top_k=50,                        # Top-k sampling for more diversity
        top_p=0.95                       # Top-p (nucleus) sampling
    )

# Decode and print the generated texts
for i, generated_sequence in enumerate(output):
    generated_text = enc.decode(generated_sequence.tolist())
    print(f"Generated text {i+1}:\n{generated_text}\n")
