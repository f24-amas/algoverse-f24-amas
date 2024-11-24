import torch
from memory_augmented import MemoryAugmentedNetwork
from preprocess import load_data, prepare_questions

# Load data
dataset = load_data('data/test.jsonl')
questions = prepare_questions(dataset)

# Determine the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move it to the appropriate device
model = MemoryAugmentedNetwork("gpt-4o-2024").to(device)

# Evaluate
correct = 0
total = len(questions)
for batch in questions:
    story = {'story_id': batch['story_id'], 'story': batch['story']}
    question = batch['question']
    answer = batch['answer_correct']

    # Move inputs to the device
    story_tensor = torch.tensor(story).to(device)  # Assuming the story can be converted into a tensor
    question_tensor = torch.tensor(question).to(device)  # Ensure the question tensor is on the same device
    
    # Forward pass (you may need to modify the `model` to accept these inputs as tensors if necessary)
    logits = model(story_tensor, question_tensor)

    # Assuming logits is a tensor of probabilities, find the predicted class
    predicted = logits.argmax().item()
    
    # Convert the answer to a numerical label (1 for "Yes", 0 for "No")
    correct += int(predicted == (1 if answer == "Yes" else 0))

# Print accuracy
print(f"Accuracy: {correct / total * 100:.2f}%")