"""
Export and Sharing System Package

Provides comprehensive export capabilities for meeting data including email delivery,
USB transfer, PDF generation, network sharing, and bulk operations.
"""

from .email_exporter import (
    EmailExporter, EmailConfig, EmailTemplate, AttachmentConfig,
    create_email_exporter, send_session_email, validate_email_config,
    create_email_template, format_email_content
)
from .usb_exporter import (
    USBExporter, USBDevice, USBConfig, TransferProgress,
    create_usb_exporter, detect_usb_drives, export_to_usb,
    format_usb_export, monitor_transfer_progress
)
from .pdf_generator import (
    PDFGenerator, PDFTemplate, PDFConfig, DocumentStyle,
    create_pdf_generator, generate_session_pdf, generate_transcript_pdf,
    generate_analysis_pdf, create_pdf_template, apply_document_style
)
from .network_sharing import (
    NetworkSharing, SMBServer, HTTPServer, SharingConfig,
    create_network_sharing, start_smb_server, start_http_server,
    configure_sharing, monitor_sharing_access
)
from .bulk_exporter import (
    BulkExporter, BulkExportConfig, ExportJob, ProgressTracker,
    create_bulk_exporter, export_multiple_sessions, create_export_job,
    track_bulk_progress, schedule_bulk_export
)
from .format_customizer import (
    FormatCustomizer, ExportTemplate, TemplateConfig, FormatOptions,
    create_format_customizer, create_export_template, customize_format,
    validate_template, apply_template_settings
)

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    """Supported export formats"""
    PDF = "pdf"
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    AUDIO = "audio"
    ZIP = "zip"

class ExportMethod(Enum):
    """Available export delivery methods"""
    EMAIL = "email"
    USB = "usb"
    SMB = "smb"
    HTTP = "http"
    LOCAL = "local"

@dataclass
class ExportConfig:
    """Configuration for export operations"""
    format: ExportFormat
    method: ExportMethod
    destination: str
    include_audio: bool = True
    include_transcript: bool = True
    include_analysis: bool = True
    custom_template: Optional[str] = None
    compression: bool = False
    password_protect: bool = False
    password: Optional[str] = None

@dataclass
class ExportResult:
    """Result of export operation"""
    success: bool
    file_path: Optional[str] = None
    destination: Optional[str] = None
    format: Optional[ExportFormat] = None
    method: Optional[ExportMethod] = None
    size_bytes: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None

class ExportManager:
    """Central export system coordinator"""
    
    def __init__(self):
        self.email_exporter = create_email_exporter()
        self.usb_exporter = create_usb_exporter()
        self.pdf_generator = create_pdf_generator()
        self.network_sharing = create_network_sharing()
        self.bulk_exporter = create_bulk_exporter()
        self.format_customizer = create_format_customizer()
        
    def export_session(self, session_id: str, config: ExportConfig) -> ExportResult:
        """Export a single session with specified configuration"""
        try:
            logger.info(f"Starting export of session {session_id} with {config.format.value} format via {config.method.value}")
            
            # Generate content based on format
            if config.format == ExportFormat.PDF:
                file_path = self.pdf_generator.generate_session_pdf(session_id, config)
            elif config.format == ExportFormat.JSON:
                file_path = self._generate_json_export(session_id, config)
            elif config.format == ExportFormat.HTML:
                file_path = self._generate_html_export(session_id, config)
            elif config.format == ExportFormat.TEXT:
                file_path = self._generate_text_export(session_id, config)
            elif config.format == ExportFormat.ZIP:
                file_path = self._generate_zip_export(session_id, config)
            else:
                raise ValueError(f"Unsupported export format: {config.format}")
            
            # Deliver via specified method
            if config.method == ExportMethod.EMAIL:
                result = self.email_exporter.send_session_email(session_id, file_path, config)
            elif config.method == ExportMethod.USB:
                result = self.usb_exporter.export_to_usb(file_path, config)
            elif config.method == ExportMethod.SMB:
                result = self.network_sharing.share_via_smb(file_path, config)
            elif config.method == ExportMethod.HTTP:
                result = self.network_sharing.share_via_http(file_path, config)
            elif config.method == ExportMethod.LOCAL:
                result = self._save_local_export(file_path, config)
            else:
                raise ValueError(f"Unsupported export method: {config.method}")
            
            logger.info(f"Export completed successfully: {result.file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Export failed for session {session_id}: {str(e)}")
            return ExportResult(
                success=False,
                error_message=str(e),
                format=config.format,
                method=config.method
            )
    
    def export_sessions(self, session_ids: List[str], config: ExportConfig) -> List[ExportResult]:
        """Export multiple sessions"""
        return self.bulk_exporter.export_multiple_sessions(session_ids, config)
    
    def get_export_formats(self) -> List[ExportFormat]:
        """Get available export formats"""
        return list(ExportFormat)
    
    def get_export_methods(self) -> List[ExportMethod]:
        """Get available export methods"""
        return list(ExportMethod)
    
    def validate_export_config(self, config: ExportConfig) -> bool:
        """Validate export configuration"""
        try:
            # Check format and method compatibility
            if config.method == ExportMethod.EMAIL and config.format == ExportFormat.AUDIO:
                logger.warning("Large audio files may fail email delivery")
            
            # Validate destination based on method
            if config.method == ExportMethod.EMAIL:
                return validate_email_config(config.destination)
            elif config.method == ExportMethod.USB:
                return self.usb_exporter.validate_usb_destination(config.destination)
            elif config.method in [ExportMethod.SMB, ExportMethod.HTTP]:
                return self.network_sharing.validate_network_destination(config.destination)
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def _generate_json_export(self, session_id: str, config: ExportConfig) -> str:
        """Generate JSON export"""
        # Implementation for JSON export
        return f"/tmp/session_{session_id}.json"
    
    def _generate_html_export(self, session_id: str, config: ExportConfig) -> str:
        """Generate HTML export"""
        # Implementation for HTML export
        return f"/tmp/session_{session_id}.html"
    
    def _generate_text_export(self, session_id: str, config: ExportConfig) -> str:
        """Generate text export"""
        # Implementation for text export
        return f"/tmp/session_{session_id}.txt"
    
    def _generate_zip_export(self, session_id: str, config: ExportConfig) -> str:
        """Generate ZIP archive export"""
        # Implementation for ZIP export
        return f"/tmp/session_{session_id}.zip"
    
    def _save_local_export(self, file_path: str, config: ExportConfig) -> ExportResult:
        """Save export to local file system"""
        # Implementation for local save
        return ExportResult(
            success=True,
            file_path=file_path,
            destination=config.destination,
            format=config.format,
            method=config.method
        )

def create_export_manager() -> ExportManager:
    """Create a new export manager instance"""
    return ExportManager()

def export_session(session_id: str, config: ExportConfig) -> ExportResult:
    """Convenience function to export a single session"""
    manager = create_export_manager()
    return manager.export_session(session_id, config)

def export_sessions(session_ids: List[str], config: ExportConfig) -> List[ExportResult]:
    """Convenience function to export multiple sessions"""
    manager = create_export_manager()
    return manager.export_sessions(session_ids, config)

def get_export_formats() -> List[ExportFormat]:
    """Get available export formats"""
    return list(ExportFormat)

def validate_export_config(config: ExportConfig) -> bool:
    """Validate export configuration"""
    manager = create_export_manager()
    return manager.validate_export_config(config)

__all__ = [
    # Core classes
    'ExportManager', 'ExportConfig', 'ExportResult', 'ExportFormat', 'ExportMethod',
    
    # Convenience functions
    'create_export_manager', 'export_session', 'export_sessions', 'get_export_formats',
    'validate_export_config',
    
    # Email exporter
    'EmailExporter', 'EmailConfig', 'EmailTemplate', 'AttachmentConfig',
    'create_email_exporter', 'send_session_email', 'validate_email_config',
    'create_email_template', 'format_email_content',
    
    # USB exporter
    'USBExporter', 'USBDevice', 'USBConfig', 'TransferProgress',
    'create_usb_exporter', 'detect_usb_drives', 'export_to_usb',
    'format_usb_export', 'monitor_transfer_progress',
    
    # PDF generator
    'PDFGenerator', 'PDFTemplate', 'PDFConfig', 'DocumentStyle',
    'create_pdf_generator', 'generate_session_pdf', 'generate_transcript_pdf',
    'generate_analysis_pdf', 'create_pdf_template', 'apply_document_style',
    
    # Network sharing
    'NetworkSharing', 'SMBServer', 'HTTPServer', 'SharingConfig',
    'create_network_sharing', 'start_smb_server', 'start_http_server',
    'configure_sharing', 'monitor_sharing_access',
    
    # Bulk exporter
    'BulkExporter', 'BulkExportConfig', 'ExportJob', 'ProgressTracker',
    'create_bulk_exporter', 'export_multiple_sessions', 'create_export_job',
    'track_bulk_progress', 'schedule_bulk_export',
    
    # Format customizer
    'FormatCustomizer', 'ExportTemplate', 'TemplateConfig', 'FormatOptions',
    'create_format_customizer', 'create_export_template', 'customize_format',
    'validate_template', 'apply_template_settings'
]