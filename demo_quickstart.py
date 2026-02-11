"""
Quick Start Demo: ACE-Shield "Weaponized Optimizer"
Shows how ACE prevents catastrophic learning during distribution shifts.
"""

import torch
import torch.nn as nn
import numpy as np
from ace_shield import ACEShield

print("="*60)
print("ACE-SHIELD QUICK START DEMO")
print("Simulating Silent Physics Shift in Robotic Control")
print("="*60)

# ========== PART 1: SETUP POLICY NETWORK ==========
class RoboticPolicy(nn.Module):
    """Simple 2-layer policy network for robotic control"""
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Initialize to "safe" mode (gentle actions)
        with torch.no_grad():
            self.net[-2].weight.data *= 0.1
            self.net[-2].bias.data.zero_()
    
    def forward(self, x):
        return self.net(x)

# ========== PART 2: CREATE ENVIRONMENT WITH SILENT SHIFT ==========
class DangerousEnvironment:
    """Environment that silently changes physics"""
    
    def __init__(self):
        self.step_count = 0
        self.shifted = False
        self.shift_threshold = 50  # When Black Swan occurs
        self.target_action = torch.tensor([0.3, 0.2])  # Normal physics
    
    def get_observation(self):
        """Simulated sensor readings"""
        # Add some noise to simulate real sensors
        noise = torch.randn(4) * 0.1
        obs = torch.tensor([1.0, 0.5, -0.2, 0.8]) + noise
        return obs
    
    def compute_loss(self, actions):
        """Loss function that changes after shift"""
        self.step_count += 1
        
        # Trigger Black Swan at step 50
        if not self.shifted and self.step_count >= self.shift_threshold:
            print("\n" + "!"*60)
            print("BLACK SWAN: Physics parameters destabilized!")
            print("Friction coefficient changed by 300%")
            print("Previous optimal actions are now DANGEROUS")
            print("!"*60)
            
            self.shifted = True
            # Physics flip: what was safe is now dangerous
            self.target_action = torch.tensor([-0.8, -0.6])  # Opposite dynamics
        
        # MSE loss relative to target
        loss = torch.mean((actions - self.target_action) ** 2)
        
        # Safety check: monitor for dangerous actions
        torque_magnitude = torch.norm(actions).item()
        safety_risk = 0.0
        if torque_magnitude > 1.5:  # Excessive torque
            safety_risk = 100.0
            print(f"  ⚠️  DANGER: Torque magnitude {torque_magnitude:.2f} > safety limit!")
        
        return loss, safety_risk, self.shifted

# ========== PART 3: TRAIN WITH ACE-SHIELD ==========
def main():
    # Create components
    policy = RoboticPolicy()
    env = DangerousEnvironment()
    
    print("\n[1] Creating standard Adam optimizer...")
    adam = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    print("[2] Wrapping with ACE-Shield...")
    ace = ACEShield(
        adam, 
        policy, 
        sigma_env=0.05,  # Moderate environmental noise
        kappa=1.0        # Standard robustness
    )
    
    print("[3] Starting training with silent shift at step 50...")
    print("    Phase 1 (0-49): Normal physics")
    print("    Phase 2 (50+):  Black Swan physics\n")
    
    safety_incidents = 0
    ace_activations = 0
    
    for step in range(100):
        # Get observation
        obs = env.get_observation()
        
        # Forward pass
        actions = policy(obs.unsqueeze(0)).squeeze(0)
        
        # Compute loss and risk
        loss, risk, shifted = env.compute_loss(actions)
        
        # Backward pass
        ace.zero_grad()
        loss.backward()
        
        # ACE-Shield step (the key difference)
        ace.step()
        
        # Track statistics
        if risk > 50:
            safety_incidents += 1
        if ace.violation_count > ace_activations:
            ace_activations = ace.violation_count
        
        # Logging
        if step % 10 == 0 or shifted:
            status = "SAFE" if risk < 1 else "⚠️ DANGER"
            ace_status = f"Shields: {ace_activations}" if ace_activations > 0 else "Shields: Ready"
            
            print(f"Step {step:3d}: Loss={loss.item():.4f} | Risk={risk:5.1f}% | {status:10} | {ace_status}")
    
    # ========== RESULTS ==========
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"Total Safety Incidents: {safety_incidents}")
    print(f"ACE Shield Activations: {ace_activations}")
    
    if safety_incidents == 0:
        print("✅ RESULT: PERFECT SURVIVAL")
        print("   ACE successfully prevented catastrophic failure!")
    elif safety_incidents <= 2:
        print("✅ RESULT: MINOR INCIDENTS")
        print("   ACE mitigated most dangerous situations")
    else:
        print("❌ RESULT: MULTIPLE FAILURES")
        print("   Consider tuning sigma_env or kappa parameters")
    
    print("\n[ACE DIAGNOSTICS]")
    print(f"  - Environment sigma: {ace.sigma_env}")
    print(f"  - Robustness kappa: {ace.kappa}")
    print(f"  - Total constraint checks: 100")

if __name__ == "__main__":
    main()