import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# -----------------------
# Hyperparameters
# -----------------------
BATCH_SIZE = 32  # Number of sequences processed in parallel
BLOCK_SIZE = 8  # Context window size for predictions
MAX_ITERS = 3000  # Number of training iterations
EVAL_INTERVAL = 300  # Interval for evaluating loss
LEARNING_RATE = 1e-2  # Learning rate for optimizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
EVAL_ITERS = 200  # Number of evaluations to average loss
SEED = 1337  # Random seed for reproducibility

torch.manual_seed(SEED)

# -----------------------
# Load Dataset
# -----------------------

# Read input text file
with open('input.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# Extract unique characters and define vocabulary
char_set = sorted(set(text_data))
VOCAB_SIZE = len(char_set)

# Create mappings: character to index and index to character
char_to_idx = {char: idx for idx, char in enumerate(char_set)}
idx_to_char = {idx: char for idx, char in enumerate(char_set)}

# Encoding function: Convert string to list of integers
def encode(text):
    return [char_to_idx[char] for char in text]

# Decoding function: Convert list of integers back to string
def decode(indices):
    return ''.join(idx_to_char[idx] for idx in indices)

# Convert text data to tensor representation
data = torch.tensor(encode(text_data), dtype=torch.long)

# Split dataset into training (90%) and validation (10%)
split_idx = int(0.9 * len(data))
train_data, val_data = data[:split_idx], data[split_idx:]

# -----------------------
# Data Loading Function
# -----------------------

def get_batch(split):
    """
    Generate a batch of training or validation data.

    Args:
        split (str): 'train' for training set, 'val' for validation set

    Returns:
        x (Tensor): Input data of shape (BATCH_SIZE, BLOCK_SIZE)
        y (Tensor): Target data of shape (BATCH_SIZE, BLOCK_SIZE)
    """
    dataset = train_data if split == 'train' else val_data
    idx = torch.randint(len(dataset) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([dataset[i:i+BLOCK_SIZE] for i in idx])
    y = torch.stack([dataset[i+1:i+BLOCK_SIZE+1] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)

# -----------------------
# Loss Estimation Function
# -----------------------

@torch.no_grad()
def estimate_loss():
    """
    Evaluate model performance on training and validation sets.

    Returns:
        dict: Averaged loss for both training and validation sets.
    """
    loss_dict = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            X_batch, Y_batch = get_batch(split)
            _, loss = model(X_batch, Y_batch)
            losses[i] = loss.item()
        loss_dict[split] = losses.mean()
    model.train()
    return loss_dict

# -----------------------
# Bigram Language Model
# -----------------------

class BigramLanguageModel(nn.Module):
    """
    A simple Bigram Language Model using an embedding table.
    """
    
    def __init__(self, vocab_size):
        super().__init__()
        # Lookup table mapping tokens to logits
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_indices, targets=None):
        """
        Forward pass through the model.

        Args:
            input_indices (Tensor): Input token indices of shape (BATCH_SIZE, BLOCK_SIZE)
            targets (Tensor, optional): Target indices for computing loss.

        Returns:
            logits (Tensor): Model predictions of shape (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)
            loss (Tensor, optional): Cross-entropy loss if targets are provided.
        """
        logits = self.embedding_table(input_indices)  # Shape: (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)

        loss = None
        if targets is not None:
            B, T, C = logits.shape  # Extract batch size, time steps, and vocab size
            logits = logits.view(B*T, C)  # Reshape for loss computation
            targets = targets.view(B*T)  # Reshape targets
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context, max_tokens):
        """
        Generate text using the trained model.

        Args:
            context (Tensor): Initial input sequence of shape (1, T)
            max_tokens (int): Maximum number of tokens to generate.

        Returns:
            Tensor: Generated token sequence.
        """
        for _ in range(max_tokens):
            logits, _ = self(context)  # Forward pass
            logits = logits[:, -1, :]  # Focus on the last token
            probs = F.softmax(logits, dim=-1)  # Convert to probability distribution
            next_token = torch.multinomial(probs, num_samples=1)  # Sample the next token
            context = torch.cat((context, next_token), dim=1)  # Append token to context
        return context

# -----------------------
# Model Training
# -----------------------

# Initialize model and move to device
model = BigramLanguageModel(VOCAB_SIZE).to(DEVICE)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for step in range(MAX_ITERS):
    
    # Evaluate loss at intervals
    if step % EVAL_INTERVAL == 0:
        loss_values = estimate_loss()
        print(f"Step {step}: Train Loss: {loss_values['train']:.4f}, Val Loss: {loss_values['val']:.4f}")

    # Fetch a batch of training data
    xb, yb = get_batch('train')

    # Compute loss and update model weights
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------
# Text Generation
# -----------------------

# Generate new text using the trained model
start_context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
generated_text = decode(model.generate(start_context, max_tokens=500)[0].tolist())

print("Generated Text:\n", generated_text)
