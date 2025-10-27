#!/usr/bin/env python3
"""
Judge Configuration System for Hackathon Grading
Allows judging committees to set custom weights for each criteria
"""

from typing import Dict, List, Optional
import json
import os

class JudgeConfig:
    """Configuration system for hackathon judging criteria weights"""
    
    def __init__(self, config_file: str = "judge_config.json"):
        self.config_file = config_file
        self.default_weights = {
            "code_analysis": 20,      # Code quality and best practices
            "architecture": 15,       # System design and structure
            "ui_ux": 15,             # User interface and experience
            "security": 10,           # Security best practices
            "innovation": 15,        # Novelty and creativity
            "functionality": 15,     # Working features and completeness
            "technical": 10,         # Technical complexity and difficulty
            "ui_ux_polish": 0,       # Visual design polish (optional)
            "learning": 0            # Learning and improvement (optional)
        }
        self.weights = self.load_config()
    
    def load_config(self) -> Dict[str, int]:
        """Load judge configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('weights', self.default_weights)
            except Exception as e:
                print(f"Warning: Could not load judge config: {e}")
                return self.default_weights
        return self.default_weights
    
    def save_config(self, weights: Dict[str, int]) -> bool:
        """Save judge configuration to file"""
        try:
            config = {
                "weights": weights,
                "total_percentage": sum(weights.values()),
                "last_updated": str(pd.Timestamp.now())
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving judge config: {e}")
            return False
    
    def validate_weights(self, weights: Dict[str, int]) -> tuple[bool, str]:
        """Validate that weights sum to 100%"""
        total = sum(weights.values())
        if total != 100:
            return False, f"Weights must sum to 100%, got {total}%"
        
        # Check for negative weights
        for criteria, weight in weights.items():
            if weight < 0:
                return False, f"Weight for {criteria} cannot be negative: {weight}%"
        
        return True, "Valid configuration"
    
    def get_weights(self) -> Dict[str, int]:
        """Get current weight configuration"""
        return self.weights.copy()
    
    def set_weights(self, weights: Dict[str, int]) -> tuple[bool, str]:
        """Set new weight configuration"""
        is_valid, message = self.validate_weights(weights)
        if is_valid:
            self.weights = weights
            self.save_config(weights)
            return True, "Configuration saved successfully"
        return False, message
    
    def get_criteria_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about each criteria"""
        return {
            "code_analysis": {
                "name": "Code Analysis",
                "description": "Code quality, patterns, and best practices",
                "weight": self.weights.get("code_analysis", 0)
            },
            "architecture": {
                "name": "Architecture Analysis", 
                "description": "System design and project structure",
                "weight": self.weights.get("architecture", 0)
            },
            "ui_ux": {
                "name": "UI/UX Analysis",
                "description": "User interface and user experience",
                "weight": self.weights.get("ui_ux", 0)
            },
            "security": {
                "name": "Security Analysis",
                "description": "Security best practices and vulnerabilities",
                "weight": self.weights.get("security", 0)
            },
            "innovation": {
                "name": "Innovation & Creativity",
                "description": "Novelty and creative use of technology",
                "weight": self.weights.get("innovation", 0)
            },
            "functionality": {
                "name": "Functionality & Completeness",
                "description": "Working features and completeness",
                "weight": self.weights.get("functionality", 0)
            },
            "technical": {
                "name": "Technical Complexity",
                "description": "Technical difficulty and architecture",
                "weight": self.weights.get("technical", 0)
            },
            "ui_ux_polish": {
                "name": "UI/UX Polish",
                "description": "Visual design and user experience polish",
                "weight": self.weights.get("ui_ux_polish", 0)
            },
            "learning": {
                "name": "Learning Agent",
                "description": "Learning from past analyses",
                "weight": self.weights.get("learning", 0)
            }
        }
    
    def get_weight_summary(self) -> str:
        """Get a summary of current weight distribution"""
        criteria_info = self.get_criteria_info()
        summary = "Current Judge Configuration:\n"
        summary += "=" * 50 + "\n"
        
        total_weight = 0
        for criteria, info in criteria_info.items():
            weight = info["weight"]
            if weight > 0:
                summary += f"• {info['name']}: {weight}%\n"
                total_weight += weight
        
        summary += f"\nTotal Weight: {total_weight}%"
        return summary

# Example usage and preset configurations
class JudgePresets:
    """Predefined judge configurations for different hackathon types"""
    
    @staticmethod
    def get_tech_focus_preset() -> Dict[str, int]:
        """Preset for technology-focused hackathons"""
        return {
            "code_analysis": 25,
            "architecture": 20,
            "technical": 20,
            "functionality": 15,
            "innovation": 10,
            "security": 5,
            "ui_ux": 5,
            "ui_ux_polish": 0,
            "learning": 0
        }
    
    @staticmethod
    def get_ui_focus_preset() -> Dict[str, int]:
        """Preset for UI/UX focused hackathons"""
        return {
            "ui_ux": 30,
            "ui_ux_polish": 25,
            "functionality": 20,
            "code_analysis": 10,
            "architecture": 10,
            "innovation": 5,
            "technical": 0,
            "security": 0,
            "learning": 0
        }
    
    @staticmethod
    def get_innovation_focus_preset() -> Dict[str, int]:
        """Preset for innovation-focused hackathons"""
        return {
            "innovation": 30,
            "functionality": 20,
            "code_analysis": 15,
            "architecture": 15,
            "technical": 10,
            "ui_ux": 5,
            "security": 5,
            "ui_ux_polish": 0,
            "learning": 0
        }
    
    @staticmethod
    def get_balanced_preset() -> Dict[str, int]:
        """Preset for balanced evaluation"""
        return {
            "code_analysis": 20,
            "architecture": 15,
            "ui_ux": 15,
            "security": 10,
            "innovation": 15,
            "functionality": 15,
            "technical": 10,
            "ui_ux_polish": 0,
            "learning": 0
        }
    
    @staticmethod
    def get_all_criteria_preset() -> Dict[str, int]:
        """Preset using all criteria"""
        return {
            "code_analysis": 15,
            "architecture": 12,
            "ui_ux": 12,
            "security": 8,
            "innovation": 15,
            "functionality": 15,
            "technical": 12,
            "ui_ux_polish": 8,
            "learning": 3
        }

if __name__ == "__main__":
    # Example usage
    config = JudgeConfig()
    
    print("🎯 Judge Configuration System")
    print("=" * 50)
    
    # Show current configuration
    print(config.get_weight_summary())
    
    # Example: Set tech focus preset
    tech_preset = JudgePresets.get_tech_focus_preset()
    success, message = config.set_weights(tech_preset)
    print(f"\nSetting tech focus preset: {message}")
    
    if success:
        print(config.get_weight_summary())
