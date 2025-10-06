#!/usr/bin/env python3
"""
Resource Usage Logger
====================
Logs GPU, CPU, RAM usage during model execution.
"""

import torch
import psutil
import time
import json
import os
from datetime import datetime
from typing import Dict, List
import threading


class ResourceLogger:
    """Monitor and log system resource usage."""

    def __init__(self, log_file: str = None, interval: float = 1.0):
        """
        Initialize resource logger.

        Args:
            log_file: Path to log file (auto-generated if None)
            interval: Sampling interval in seconds
        """
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/resource_usage_{timestamp}.json"

        self.log_file = log_file
        self.interval = interval
        self.samples = []
        self.is_running = False
        self.thread = None
        self.start_time = None

        # Create logs directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def get_gpu_info(self) -> Dict:
        """Get GPU memory and utilization info."""
        gpu_info = {}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = props.total_memory / (1024**3)

                gpu_info[f'gpu_{i}'] = {
                    'name': props.name,
                    'allocated_gb': round(allocated, 2),
                    'reserved_gb': round(reserved, 2),
                    'total_gb': round(total, 2),
                    'utilization_pct': round((allocated / total * 100), 1) if total > 0 else 0
                }

        return gpu_info

    def get_cpu_ram_info(self) -> Dict:
        """Get CPU and RAM usage info."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()

        return {
            'cpu_percent': round(cpu_percent, 1),
            'cpu_count': psutil.cpu_count(),
            'ram_used_gb': round(ram.used / (1024**3), 2),
            'ram_total_gb': round(ram.total / (1024**3), 2),
            'ram_percent': round(ram.percent, 1)
        }

    def get_process_info(self) -> Dict:
        """Get current process resource usage."""
        process = psutil.Process()

        return {
            'pid': process.pid,
            'cpu_percent': round(process.cpu_percent(), 1),
            'memory_mb': round(process.memory_info().rss / (1024**2), 2),
            'num_threads': process.num_threads()
        }

    def sample_resources(self) -> Dict:
        """Take a single resource sample."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        sample = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': round(elapsed, 2),
            'gpu': self.get_gpu_info(),
            'system': self.get_cpu_ram_info(),
            'process': self.get_process_info()
        }

        return sample

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            sample = self.sample_resources()
            self.samples.append(sample)
            time.sleep(self.interval)

    def start(self):
        """Start monitoring in background."""
        if self.is_running:
            return

        self.is_running = True
        self.start_time = time.time()
        self.samples = []

        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()

        print(f"📊 Resource monitoring started (interval: {self.interval}s)")

    def stop(self):
        """Stop monitoring."""
        if not self.is_running:
            return

        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)

        print(f"📊 Resource monitoring stopped ({len(self.samples)} samples)")

    def save_log(self, metadata: Dict = None):
        """Save resource log to file."""
        log_data = {
            'metadata': metadata or {},
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            'duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'sample_count': len(self.samples),
            'sample_interval': self.interval,
            'samples': self.samples,
            'summary': self.get_summary()
        }

        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"💾 Resource log saved to: {self.log_file}")

    def get_summary(self) -> Dict:
        """Get summary statistics from samples."""
        if not self.samples:
            return {}

        # Extract GPU metrics
        gpu_summary = {}
        for gpu_id in self.samples[0]['gpu'].keys():
            utilizations = [s['gpu'][gpu_id]['utilization_pct'] for s in self.samples]
            allocated = [s['gpu'][gpu_id]['allocated_gb'] for s in self.samples]

            gpu_summary[gpu_id] = {
                'peak_utilization_pct': round(max(utilizations), 1),
                'avg_utilization_pct': round(sum(utilizations) / len(utilizations), 1),
                'peak_memory_gb': round(max(allocated), 2),
                'avg_memory_gb': round(sum(allocated) / len(allocated), 2)
            }

        # Extract CPU/RAM metrics
        cpu_usage = [s['system']['cpu_percent'] for s in self.samples]
        ram_usage = [s['system']['ram_percent'] for s in self.samples]

        system_summary = {
            'cpu_peak_pct': round(max(cpu_usage), 1),
            'cpu_avg_pct': round(sum(cpu_usage) / len(cpu_usage), 1),
            'ram_peak_pct': round(max(ram_usage), 1),
            'ram_avg_pct': round(sum(ram_usage) / len(ram_usage), 1)
        }

        return {
            'gpu': gpu_summary,
            'system': system_summary
        }

    def print_summary(self):
        """Print resource usage summary."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("📊 RESOURCE USAGE SUMMARY")
        print("=" * 70)

        if 'gpu' in summary:
            print("\n🎮 GPU Usage:")
            for gpu_id, stats in summary['gpu'].items():
                print(f"  {gpu_id.upper()}:")
                print(f"    Peak: {stats['peak_memory_gb']}GB ({stats['peak_utilization_pct']}%)")
                print(f"    Avg:  {stats['avg_memory_gb']}GB ({stats['avg_utilization_pct']}%)")

        if 'system' in summary:
            # Get system specs from first sample
            cpu_count = self.samples[0]['system']['cpu_count'] if self.samples else psutil.cpu_count()
            ram_total_gb = self.samples[0]['system']['ram_total_gb'] if self.samples else round(psutil.virtual_memory().total / (1024**3), 2)

            # Calculate actual usage
            cpu_peak_cores = round((summary['system']['cpu_peak_pct'] / 100) * cpu_count, 1)
            cpu_avg_cores = round((summary['system']['cpu_avg_pct'] / 100) * cpu_count, 1)
            ram_peak_gb = round((summary['system']['ram_peak_pct'] / 100) * ram_total_gb, 1)
            ram_avg_gb = round((summary['system']['ram_avg_pct'] / 100) * ram_total_gb, 1)

            print(f"\n💻 System Usage:")
            print(f"  CPU:  Peak {cpu_peak_cores}/{cpu_count} cores ({summary['system']['cpu_peak_pct']}%) | Avg {cpu_avg_cores}/{cpu_count} cores ({summary['system']['cpu_avg_pct']}%)")
            print(f"  RAM:  Peak {ram_peak_gb}/{ram_total_gb}GB ({summary['system']['ram_peak_pct']}%) | Avg {ram_avg_gb}/{ram_total_gb}GB ({summary['system']['ram_avg_pct']}%)")

        print("=" * 70)

    def log_event(self, event_name: str, metadata: Dict = None):
        """Log a specific event with current resource snapshot."""
        sample = self.sample_resources()
        sample['event'] = event_name
        sample['event_metadata'] = metadata or {}
        self.samples.append(sample)

        print(f"📍 Event logged: {event_name}")


class ResourceMonitor:
    """Context manager for easy resource monitoring."""

    def __init__(self, model_name: str = None, log_file: str = None, interval: float = 1.0):
        self.logger = ResourceLogger(log_file, interval)
        self.model_name = model_name

    def __enter__(self):
        self.logger.start()
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.stop()

        # Save with metadata
        metadata = {
            'model_name': self.model_name,
            'success': exc_type is None,
            'error': str(exc_val) if exc_val else None
        }

        self.logger.save_log(metadata)
        self.logger.print_summary()


# Convenience function
def monitor_resources(model_name: str = None, log_file: str = None, interval: float = 1.0):
    """
    Convenience context manager for resource monitoring.

    Usage:
        with monitor_resources("facebook/opt-30b"):
            # Your code here
            model.generate(...)
    """
    return ResourceMonitor(model_name, log_file, interval)


if __name__ == "__main__":
    # Test the logger
    print("Testing resource logger...")

    with monitor_resources("test-model") as logger:
        logger.log_event("model_loading")
        time.sleep(2)
        logger.log_event("inference_start")
        time.sleep(3)
        logger.log_event("inference_complete")

    print("\n✅ Test complete!")
