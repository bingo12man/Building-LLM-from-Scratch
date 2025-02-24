import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# -----------------------
# Hyperparameters
# -----------------------
BATCH_SIZE = 64       # Number of sequences processed in parallel
BLOCK_SIZE = 256      # Context window size for predictions
MAX_ITERS = 5000      # Number of training iterations
EVAL_INTERVAL = 500   # Interval for evaluating loss
LEARNING_RATE = 3e-4  # Learning rate for optimizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
EVAL_ITERS = 200      # Number of evaluations for loss estimation
N_EMBED = 384         # Embedding dimension
N_HEAD = 6            # Number of attention heads
N_LAYER = 6           # Number of Transformer blocks
DROPOUT = 0.2         # Dropout rate for regularization

torch.manual_seed(1337)

# -----------------------
# Load Dataset
# -----------------------

# Read input text file
with open('input.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# Extract unique characters and define vocabulary
char_set = sorted(set(text_data))
VOCAB_SIZE = len(char_set)

# Create character-to-index and index-to-character mappings
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
# Data Loader
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
# Transformer Components
# -----------------------

class SelfAttentionHead(nn.Module):
    """ One attention head for self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.tril = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) * k.shape[-1]**-0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        v = self.value(x)
        return attention_weights @ v

class MultiHeadAttention(nn.Module):
    """ Multi-head self-attention """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    """ Feedforward network with non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.network(x)

class TransformerBlock(nn.Module):
    """ Transformer block: Multi-head attention + FeedForward """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.feedforward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x

# -----------------------
# GPT Model
# -----------------------

class GPTLanguageModel(nn.Module):
    """ Transformer-based GPT model """

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[TransformerBlock(N_EMBED, N_HEAD) for _ in range(N_LAYER)])
        self.layer_norm = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Generate new text based on input context """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# -----------------------
# Model Training
# -----------------------

model = GPTLanguageModel().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {step}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    optimizer.zero_grad()
    _, loss = model(xb, yb)
    loss.backward()
    optimizer.step()

# -----------------------
# Generate Text
# -----------------------

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
