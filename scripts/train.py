# train.py (same applies to evaluate.py)
import torch
from torch.optim import Adam
from memory_augmented import MemoryAugmentedNetwork
from preprocess import load_data, prepare_questions

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data from all four files
stories, memories, behaviors, judgments = load_data(
    'data/story.jsonl', 'data/memory.jsonl', 'data/behavior.jsonl', 'data/judgment.jsonl'
)

# Prepare the questions
questions = prepare_questions(stories, memories, behaviors, judgments)

# Initialize model and optimizer
model = MemoryAugmentedNetwork("gpt-4o-2024", device).to(device)
optimizer = Adam(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):  # Number of epochs
    for batch in questions:
        story = {'story_id': batch['story_id'], 'story': batch['story']}
        question = batch['question']
        label = 1 if batch['answer_correct'] == "Yes" else 0

        optimizer.zero_grad()
        logits = model(story, question)
        loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([label]).to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")