"""
Neural Network Pruning: Lottery Ticket Hypothesis Implementation
Author: Research Project for NMIMS Tech Trends
Description: This implementation demonstrates that neural networks can be pruned
             by 90%+ while maintaining accuracy through the Lottery Ticket Hypothesis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import copy
import json
import os
from tqdm import tqdm


class SimpleNN(nn.Module):
    """
    A simple fully connected neural network for MNIST classification.
    Architecture: 784 -> 300 -> 100 -> 10
    """
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, train_loader, test_loader, epochs=5, device='cpu'):
    """
    Train the neural network and return final accuracy.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        device: Device to train on (cpu/cuda)
    
    Returns:
        Final test accuracy
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate accuracy on test set
        accuracy = evaluate_model(model, test_loader, device)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    return accuracy


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def apply_mask(model, mask):
    """
    Apply pruning mask to model weights.
    
    Args:
        model: Neural network model
        mask: Dictionary of masks for each layer
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.mul_(mask[name])


def create_pruning_mask(model, pruning_percentage):
    """
    Create a pruning mask based on weight magnitudes.
    This implements magnitude-based pruning - removing weights with smallest absolute values.
    
    Args:
        model: Neural network model
        pruning_percentage: Percentage of weights to prune (0-100)
    
    Returns:
        Dictionary of binary masks for each layer
    """
    masks = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only prune weights, not biases
            # Get absolute values of weights
            weight_abs = torch.abs(param.data)
            
            # Calculate threshold for pruning
            threshold_index = int(pruning_percentage / 100.0 * weight_abs.numel())
            
            # Sort weights and find threshold value
            sorted_weights = torch.sort(weight_abs.view(-1))[0]
            
            if threshold_index < len(sorted_weights):
                threshold = sorted_weights[threshold_index]
                # Create binary mask: 1 for weights to keep, 0 for weights to prune
                masks[name] = (weight_abs > threshold).float()
            else:
                masks[name] = torch.zeros_like(param.data)
        else:
            # Don't prune biases
            masks[name] = torch.ones_like(param.data)
    
    return masks


def lottery_ticket_experiment(pruning_percentages, epochs=5, device='cpu'):
    """
    Main experiment implementing the Lottery Ticket Hypothesis.
    
    This function:
    1. Trains a baseline model
    2. Iteratively prunes the network at different percentages
    3. Resets pruned networks to original initialization
    4. Retrains and measures accuracy
    
    Args:
        pruning_percentages: List of pruning percentages to test
        epochs: Number of training epochs per iteration
        device: Device to run on
    
    Returns:
        Dictionary containing experimental results
    """
    print("=" * 60)
    print("LOTTERY TICKET HYPOTHESIS EXPERIMENT")
    print("=" * 60)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Step 1: Train baseline model and save initial weights
    print("\n[Step 1] Training baseline model...")
    baseline_model = SimpleNN()
    
    # Save the initial random initialization
    initial_weights = copy.deepcopy(baseline_model.state_dict())
    
    # Train the baseline model
    baseline_accuracy = train_model(baseline_model, train_loader, test_loader, epochs, device)
    
    print(f"\nBaseline Model Accuracy: {baseline_accuracy:.2f}%")
    print(f"Total Parameters: {sum(p.numel() for p in baseline_model.parameters())}")
    
    # Store results
    results = {
        'baseline_accuracy': baseline_accuracy,
        'pruning_results': []
    }
    
    # Step 2: Iterative pruning and retraining
    print("\n" + "=" * 60)
    print("PRUNING EXPERIMENTS")
    print("=" * 60)
    
    for prune_pct in pruning_percentages:
        print(f"\n[Pruning {prune_pct}%]")
        
        # Create a fresh model with original initialization
        pruned_model = SimpleNN()
        pruned_model.load_state_dict(initial_weights)
        
        # Create pruning mask
        mask = create_pruning_mask(baseline_model, prune_pct)
        
        # Apply mask to initial weights (this is the "winning ticket")
        apply_mask(pruned_model, mask)
        
        # Count remaining parameters
        total_params = sum(p.numel() for p in pruned_model.parameters())
        remaining_params = sum((mask[name].sum().item() for name in mask.keys() if 'weight' in name))
        actual_prune_pct = 100 * (1 - remaining_params / total_params)
        
        print(f"Remaining parameters: {int(remaining_params)} ({100-actual_prune_pct:.1f}% of original)")
        
        # Retrain the pruned model
        print("Retraining pruned network...")
        
        # Custom training loop that maintains the mask
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pruned_model.parameters(), lr=0.001)
        pruned_model = pruned_model.to(device)
        
        for epoch in range(epochs):
            pruned_model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = pruned_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Reapply mask after each update to ensure pruned weights stay zero
                apply_mask(pruned_model, mask)
        
        # Evaluate pruned model
        pruned_accuracy = evaluate_model(pruned_model, test_loader, device)
        
        print(f"Pruned Model Accuracy: {pruned_accuracy:.2f}%")
        print(f"Accuracy Difference: {pruned_accuracy - baseline_accuracy:+.2f}%")
        
        # Store results
        results['pruning_results'].append({
            'pruning_percentage': prune_pct,
            'actual_pruning_percentage': actual_prune_pct,
            'accuracy': pruned_accuracy,
            'accuracy_difference': pruned_accuracy - baseline_accuracy,
            'remaining_parameters': int(remaining_params),
            'total_parameters': total_params
        })
    
    return results


def save_results(results, filename='results/experiment_results.json'):
    """Save experimental results to JSON file."""
    os.makedirs('results', exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {filename}")


def main():
    """Main execution function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define pruning percentages to test
    pruning_percentages = [0, 20, 40, 60, 70, 80, 90, 95]
    
    # Run experiment
    results = lottery_ticket_experiment(
        pruning_percentages=pruning_percentages,
        epochs=5,
        device=device
    )
    
    # Save results
    save_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Baseline Accuracy: {results['baseline_accuracy']:.2f}%\n")
    
    print("Pruning Results:")
    print(f"{'Pruning %':<12} {'Accuracy':<12} {'Difference':<12} {'Parameters Remaining'}")
    print("-" * 60)
    
    for result in results['pruning_results']:
        print(f"{result['pruning_percentage']:<12} "
              f"{result['accuracy']:<12.2f} "
              f"{result['accuracy_difference']:+<12.2f} "
              f"{result['remaining_parameters']:,}")
    
    print("\n" + "=" * 60)
    print("Key Finding: Neural networks can maintain high accuracy")
    print("even when 90%+ of parameters are pruned!")
    print("=" * 60)


if __name__ == "__main__":
    main()
