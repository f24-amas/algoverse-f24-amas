import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class MemoryAugmentedNetwork(torch.nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device
        self.memory = {}

    def store_memory(self, story_id, key_info):
        """Store key information about the story."""
        self.memory[story_id] = key_info

    def query_memory(self, story_id, question):
        """Retrieve memory related to the story."""
        return self.memory.get(story_id, "")

    def forward(self, story, question):
        """Run the model on the story and question, using memory."""
        memory = self.query_memory(story['story_id'], question)
        combined_input = f"{story['story']} Memory: {memory} Question: {question}"
        
        # Initialize tokenizer and encode inputs
        tokenizer = AutoTokenizer.from_pretrained("gpt-4o-2024")
        inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)
        
        # Move inputs to the correct device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Forward pass
        outputs = self.base_model(**inputs)
        return outputs.logits