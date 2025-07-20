import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from logger import get_logger

class TrainingSummary:
    """
    Class to track and visualize training metrics
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics = {
            "loss": [],
            "learning_rate": [],
            "steps": [],
            "epoch": []
        }
        self.start_time = datetime.now()
        self.summary_file = os.path.join(output_dir, "training_summary.json")
        
        # Create metrics directory
        self.metrics_dir = os.path.join(output_dir, "metrics")
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)
    
    def add_metric(self, step, epoch, loss, learning_rate):
        """Add a training metric at a specific step"""
        self.metrics["steps"].append(step)
        self.metrics["epoch"].append(epoch)
        self.metrics["loss"].append(loss)
        self.metrics["learning_rate"].append(learning_rate)
        
        # Save metrics after each update
        self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        try:
            with open(self.summary_file, 'w') as f:
                json.dump({
                    "metrics": self.metrics,
                    "start_time": self.start_time.isoformat(),
                    "last_update": datetime.now().isoformat(),
                    "total_steps": len(self.metrics["steps"])
                }, f, indent=2)
        except Exception as e:
            get_logger().error(f"Error saving metrics: {str(e)}")
    
    def plot_loss_curve(self):
        """Plot and save the loss curve"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics["steps"], self.metrics["loss"])
            plt.title("Training Loss")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.grid(True)
            
            # Save the plot
            loss_plot_path = os.path.join(self.metrics_dir, "loss_curve.png")
            plt.savefig(loss_plot_path)
            plt.close()
            
            get_logger().info(f"Loss curve saved to {loss_plot_path}")
        except Exception as e:
            get_logger().error(f"Error plotting loss curve: {str(e)}")
    
    def plot_learning_rate(self):
        """Plot and save the learning rate curve"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics["steps"], self.metrics["learning_rate"])
            plt.title("Learning Rate Schedule")
            plt.xlabel("Steps")
            plt.ylabel("Learning Rate")
            plt.grid(True)
            
            # Save the plot
            lr_plot_path = os.path.join(self.metrics_dir, "learning_rate.png")
            plt.savefig(lr_plot_path)
            plt.close()
            
            get_logger().info(f"Learning rate curve saved to {lr_plot_path}")
        except Exception as e:
            get_logger().error(f"Error plotting learning rate curve: {str(e)}")
    
    def generate_training_summary(self):
        """Generate a comprehensive training summary"""
        try:
            # Calculate training duration
            duration = datetime.now() - self.start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Calculate summary statistics
            avg_loss = np.mean(self.metrics["loss"]) if self.metrics["loss"] else 0
            min_loss = np.min(self.metrics["loss"]) if self.metrics["loss"] else 0
            min_loss_step = self.metrics["steps"][np.argmin(self.metrics["loss"])] if self.metrics["loss"] else 0
            
            # Create summary report
            summary = {
                "training_duration": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
                "total_steps": len(self.metrics["steps"]),
                "total_epochs": max(self.metrics["epoch"]) if self.metrics["epoch"] else 0,
                "average_loss": avg_loss,
                "minimum_loss": min_loss,
                "min_loss_step": min_loss_step,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
            }
            
            # Save summary report
            summary_path = os.path.join(self.metrics_dir, "training_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            get_logger().info(f"Training summary saved to {summary_path}")
            
            # Generate plots
            self.plot_loss_curve()
            self.plot_learning_rate()
            
            return summary
        except Exception as e:
            get_logger().error(f"Error generating training summary: {str(e)}")
            return {}
