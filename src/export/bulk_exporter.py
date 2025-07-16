"""
Bulk Export System

Multi-session export operations with progress tracking and job management.
"""

import logging
import threading
import time
import queue
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
from pathlib import Path
import uuid
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Export job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class JobPriority(Enum):
    """Export job priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class ExportJob:
    """Export job definition"""
    job_id: str
    session_ids: List[str]
    export_config: Any  # ExportConfig from parent module
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_session: Optional[str] = None
    results: List[Any] = field(default_factory=list)  # ExportResult objects
    error_message: Optional[str] = None
    estimated_duration: Optional[int] = None  # seconds
    
@dataclass
class BulkExportConfig:
    """Bulk export configuration"""
    max_concurrent_jobs: int = 3
    max_sessions_per_job: int = 50
    retry_failed_sessions: bool = True
    max_retries: int = 3
    job_timeout: int = 3600  # 1 hour
    save_job_history: bool = True
    cleanup_old_jobs: bool = True
    job_history_days: int = 30
    progress_update_interval: float = 1.0  # seconds

@dataclass
class ProgressTracker:
    """Progress tracking information"""
    job_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    current_job: Optional[str] = None
    current_session: Optional[str] = None
    overall_progress: float = 0.0
    job_progress: float = 0.0
    estimated_time_remaining: Optional[int] = None
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

class BulkExporter:
    """Multi-session batch export operations"""
    
    def __init__(self, config: BulkExportConfig):
        self.config = config
        self.jobs: Dict[str, ExportJob] = {}
        self.job_queue = queue.PriorityQueue()
        self.worker_threads: List[threading.Thread] = []
        self.running = False
        self.progress_callbacks: List[Callable[[ProgressTracker], None]] = []
        self.job_lock = threading.Lock()
        
    def start_workers(self):
        """Start worker threads for processing jobs"""
        if self.running:
            return
            
        self.running = True
        
        for i in range(self.config.max_concurrent_jobs):
            worker = threading.Thread(target=self._worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Started {self.config.max_concurrent_jobs} export workers")
    
    def stop_workers(self):
        """Stop all worker threads"""
        self.running = False
        
        # Add poison pills to wake up workers
        for _ in self.worker_threads:
            self.job_queue.put((0, None))
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5)
        
        self.worker_threads.clear()
        logger.info("Stopped export workers")
    
    def _worker_thread(self, worker_id: int):
        """Worker thread for processing export jobs"""
        logger.info(f"Export worker {worker_id} started")
        
        while self.running:
            try:
                # Get job from queue (blocks until available)
                priority, job_id = self.job_queue.get(timeout=1)
                
                if job_id is None:  # Poison pill
                    break
                
                # Process job
                self._process_job(job_id, worker_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
        
        logger.info(f"Export worker {worker_id} stopped")
    
    def _process_job(self, job_id: str, worker_id: int):
        """Process a single export job"""
        with self.job_lock:
            if job_id not in self.jobs:
                logger.warning(f"Job {job_id} not found")
                return
            
            job = self.jobs[job_id]
            if job.status != JobStatus.PENDING:
                logger.warning(f"Job {job_id} not in pending status: {job.status}")
                return
            
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
        
        logger.info(f"Worker {worker_id} processing job {job_id} with {len(job.session_ids)} sessions")
        
        try:
            # Import export manager here to avoid circular imports
            from . import create_export_manager
            export_manager = create_export_manager()
            
            completed_count = 0
            failed_count = 0
            
            for i, session_id in enumerate(job.session_ids):
                if not self.running or job.status == JobStatus.CANCELLED:
                    break
                
                with self.job_lock:
                    job.current_session = session_id
                    job.progress = i / len(job.session_ids)
                
                # Update progress
                self._update_progress(job_id)
                
                try:
                    # Export session
                    result = export_manager.export_session(session_id, job.export_config)
                    
                    with self.job_lock:
                        job.results.append(result)
                    
                    if result.success:
                        completed_count += 1
                        logger.debug(f"Exported session {session_id} successfully")
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to export session {session_id}: {result.error_message}")
                        
                        # Retry if configured
                        if self.config.retry_failed_sessions and failed_count <= self.config.max_retries:
                            logger.info(f"Retrying session {session_id}")
                            time.sleep(1)  # Brief delay before retry
                            continue
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error exporting session {session_id}: {str(e)}")
                
                # Small delay between sessions
                time.sleep(0.1)
            
            # Job completed
            with self.job_lock:
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.COMPLETED if failed_count == 0 else JobStatus.FAILED
                    job.completed_at = datetime.now()
                    job.progress = 1.0
                    job.current_session = None
                    
                    if failed_count > 0:
                        job.error_message = f"{failed_count} sessions failed to export"
            
            # Final progress update
            self._update_progress(job_id)
            
            logger.info(f"Job {job_id} completed: {completed_count} success, {failed_count} failed")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed with error: {str(e)}")
            with self.job_lock:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = str(e)
    
    def export_multiple_sessions(self, session_ids: List[str], export_config: Any, 
                                priority: JobPriority = JobPriority.NORMAL) -> str:
        """Schedule export of multiple sessions"""
        # Validate session list
        if not session_ids:
            raise ValueError("Session list cannot be empty")
        
        if len(session_ids) > self.config.max_sessions_per_job:
            raise ValueError(f"Too many sessions: {len(session_ids)} > {self.config.max_sessions_per_job}")
        
        # Create job
        job_id = str(uuid.uuid4())
        job = ExportJob(
            job_id=job_id,
            session_ids=session_ids.copy(),
            export_config=export_config,
            priority=priority,
            estimated_duration=len(session_ids) * 30  # Estimate 30 seconds per session
        )
        
        with self.job_lock:
            self.jobs[job_id] = job
        
        # Queue job (priority queue uses negative value for higher priority)
        priority_value = -priority.value
        self.job_queue.put((priority_value, job_id))
        
        logger.info(f"Queued export job {job_id} with {len(session_ids)} sessions")
        
        # Start workers if not already running
        if not self.running:
            self.start_workers()
        
        return job_id
    
    def create_export_job(self, session_ids: List[str], export_config: Any, 
                         priority: JobPriority = JobPriority.NORMAL) -> ExportJob:
        """Create export job without queueing"""
        job_id = str(uuid.uuid4())
        job = ExportJob(
            job_id=job_id,
            session_ids=session_ids.copy(),
            export_config=export_config,
            priority=priority
        )
        
        with self.job_lock:
            self.jobs[job_id] = job
        
        return job
    
    def schedule_job(self, job: ExportJob) -> bool:
        """Schedule existing job for execution"""
        if job.job_id not in self.jobs:
            logger.error(f"Job {job.job_id} not found")
            return False
        
        if job.status != JobStatus.PENDING:
            logger.error(f"Job {job.job_id} not in pending status")
            return False
        
        # Queue job
        priority_value = -job.priority.value
        self.job_queue.put((priority_value, job.job_id))
        
        # Start workers if not already running
        if not self.running:
            self.start_workers()
        
        return True
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        with self.job_lock:
            if job_id not in self.jobs:
                logger.warning(f"Job {job_id} not found")
                return False
            
            job = self.jobs[job_id]
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                logger.warning(f"Job {job_id} cannot be cancelled: {job.status}")
                return False
            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def get_job_status(self, job_id: str) -> Optional[ExportJob]:
        """Get job status"""
        with self.job_lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[ExportJob]:
        """Get all jobs"""
        with self.job_lock:
            return list(self.jobs.values())
    
    def get_active_jobs(self) -> List[ExportJob]:
        """Get active jobs (pending or running)"""
        with self.job_lock:
            return [job for job in self.jobs.values() 
                   if job.status in [JobStatus.PENDING, JobStatus.RUNNING]]
    
    def track_bulk_progress(self) -> ProgressTracker:
        """Track overall bulk export progress"""
        with self.job_lock:
            total_jobs = len(self.jobs)
            completed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED])
            failed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.FAILED])
            running_jobs = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING]
            
            current_job = running_jobs[0].job_id if running_jobs else None
            current_session = running_jobs[0].current_session if running_jobs else None
            
            overall_progress = (completed_jobs + failed_jobs) / total_jobs if total_jobs > 0 else 0
            job_progress = running_jobs[0].progress if running_jobs else 0
            
            return ProgressTracker(
                job_id=current_job or "none",
                total_jobs=total_jobs,
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                current_job=current_job,
                current_session=current_session,
                overall_progress=overall_progress,
                job_progress=job_progress
            )
    
    def _update_progress(self, job_id: str):
        """Update progress and notify callbacks"""
        progress = self.track_bulk_progress()
        
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback error: {str(e)}")
    
    def add_progress_callback(self, callback: Callable[[ProgressTracker], None]):
        """Add progress callback"""
        self.progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[ProgressTracker], None]):
        """Remove progress callback"""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    def cleanup_old_jobs(self):
        """Clean up old completed jobs"""
        if not self.config.cleanup_old_jobs:
            return
        
        cutoff_time = datetime.now() - timedelta(days=self.config.job_history_days)
        
        with self.job_lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                    job.completed_at and job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def save_job_history(self, file_path: str):
        """Save job history to file"""
        try:
            with self.job_lock:
                jobs_data = []
                for job in self.jobs.values():
                    job_dict = {
                        'job_id': job.job_id,
                        'session_count': len(job.session_ids),
                        'priority': job.priority.name,
                        'status': job.status.name,
                        'created_at': job.created_at.isoformat(),
                        'started_at': job.started_at.isoformat() if job.started_at else None,
                        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                        'progress': job.progress,
                        'error_message': job.error_message,
                        'results_count': len(job.results)
                    }
                    jobs_data.append(job_dict)
            
            with open(file_path, 'w') as f:
                json.dump(jobs_data, f, indent=2)
            
            logger.info(f"Saved job history to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save job history: {str(e)}")

def create_bulk_exporter(config: Optional[BulkExportConfig] = None) -> BulkExporter:
    """Create bulk exporter with default or provided configuration"""
    if config is None:
        config = BulkExportConfig()
    
    return BulkExporter(config)

def export_multiple_sessions(session_ids: List[str], export_config: Any, 
                           priority: JobPriority = JobPriority.NORMAL) -> str:
    """Convenience function to export multiple sessions"""
    exporter = create_bulk_exporter()
    return exporter.export_multiple_sessions(session_ids, export_config, priority)

def create_export_job(session_ids: List[str], export_config: Any, 
                     priority: JobPriority = JobPriority.NORMAL) -> ExportJob:
    """Convenience function to create export job"""
    exporter = create_bulk_exporter()
    return exporter.create_export_job(session_ids, export_config, priority)

def track_bulk_progress(exporter: BulkExporter) -> ProgressTracker:
    """Convenience function to track bulk progress"""
    return exporter.track_bulk_progress()

def schedule_bulk_export(session_ids: List[str], export_config: Any, 
                        schedule_time: datetime) -> str:
    """Schedule bulk export for later execution"""
    # This would integrate with a scheduling system
    # For now, just create immediate job
    exporter = create_bulk_exporter()
    return exporter.export_multiple_sessions(session_ids, export_config)