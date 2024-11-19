from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pickle

# Load the data
# comments = pd.read_pickle('worldnews_comments.pkl')
# comments['score'] = pd.to_numeric(comments['score'], errors='coerce')
# comments['body'] = comments['body'].str[:512]

# # Filter and preprocess comments
# comments = comments[comments['score'] > 250]['body'].astype(str)

# # Load pre-trained tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()  # Set to evaluation mode

# # Use GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# # Reduce to 16-bit precision for faster processing
# model.half()

# # Tokenize the comments
# encodings = tokenizer(
#     list(comments),
#     padding=True,
#     truncation=True,
#     max_length=512,
#     return_tensors="pt"
# )

# # Create DataLoader for batching
# batch_size = 100  # Adjust batch size for your GPU memory
# dataloader = DataLoader(
#     list(zip(encodings['input_ids'], encodings['attention_mask'])), 
#     batch_size=batch_size
# )

# # Process embeddings and save after each batch
output_file = '16bit_embeddings.pt'
# with torch.no_grad():
#     for i, (input_ids, attention_mask) in enumerate(tqdm(dataloader, desc="Generating Embeddings")):
#         # Move data to GPU
#         input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

#         # Generate embeddings
#         batch_embeddings = model(input_ids, attention_mask=attention_mask)[0].cpu()  # Move results back to CPU
        
#         # Save embeddings for the batch
#         torch.save(batch_embeddings, f"embeddings/embeddings_16bit_batch_{i}.pt")

# Combine all batch files if needed (optional)
import glob
batch_files = sorted(glob.glob('embeddings/embeddings_16bit_batch_*.pt'))

batch_files_1 = batch_files[:150]
batch_files_2 = batch_files[150:300]
batch_files_3 = batch_files[300:450]
batch_files_4 = batch_files[450:]

embeddings_1 = []
for f in batch_files_1:
    batch_embeddings = torch.load(f)
    print(f"Loaded {f} with shape {batch_embeddings.shape}")
    embeddings_1.append(batch_embeddings)

with open('embeddings/embeddings_1.pkl', 'wb') as f:
    pickle.dump(embeddings_1, f)

print("Saved embeddings_1")

embeddings_2 = []

for f in batch_files_2:
    batch_embeddings = torch.load(f)
    print(f"Loaded {f} with shape {batch_embeddings.shape}")
    embeddings_2.append(batch_embeddings)

with open('embeddings/embeddings_2.pkl', 'wb') as f:
    pickle.dump(embeddings_2, f)

print("Saved embeddings_2")

embeddings_3 = []

for f in batch_files_3:
    batch_embeddings = torch.load(f)
    print(f"Loaded {f} with shape {batch_embeddings.shape}")
    embeddings_3.append(batch_embeddings)

with open('embeddings/embeddings_3.pkl', 'wb') as f:
    pickle.dump(embeddings_3, f)

print("Saved embeddings_3")

embeddings_4 = []

for f in batch_files_4:
    batch_embeddings = torch.load(f)
    print(f"Loaded {f} with shape {batch_embeddings.shape}")
    embeddings_4.append(batch_embeddings)

with open('embeddings/embeddings_4.pkl', 'wb') as f:
    pickle.dump(embeddings_4, f)

print("Saved embeddings_4")