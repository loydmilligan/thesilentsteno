# Export Module Documentation

## Module Overview

The Export module provides comprehensive export and sharing capabilities for The Silent Steno meeting data. It supports multiple export formats (PDF, text, JSON, HTML, audio), delivery methods (email, USB, network sharing), and advanced features like bulk operations, template customization, and progress tracking. The module is designed for flexibility and reliability in data export workflows.

## Dependencies

### External Dependencies
- `reportlab` - PDF generation
- `jinja2` - Template rendering
- `smtplib` - Email sending
- `email` - Email message construction
- `weasyprint` - HTML to PDF conversion
- `requests` - HTTP operations
- `paramiko` - SSH/SFTP operations
- `threading` - Multi-threading for concurrent operations
- `queue` - Thread-safe queues
- `json` - JSON serialization
- `zipfile` - Archive creation
- `pathlib` - Path operations
- `datetime` - Date/time operations
- `logging` - Logging system
- `dataclasses` - Data structures
- `enum` - Enumerations
- `typing` - Type hints
- `hashlib` - Hash functions
- `shutil` - File operations
- `tempfile` - Temporary file operations
- `subprocess` - Process execution
- `socket` - Network operations
- `http.server` - HTTP server
- `socketserver` - Socket server
- `urllib.parse` - URL parsing

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.logging` - Logging system
- `src.core.monitoring` - Performance monitoring
- `src.data.models` - Data models
- `src.data.database` - Database access

## File Documentation

### 1. `__init__.py`

**Purpose**: Main export system coordinator providing unified access to all export capabilities.

#### Classes

##### `ExportManager`
Central export management system coordinating all export operations.

**Attributes:**
- `config: dict` - Export configuration
- `bulk_exporter: BulkExporter` - Bulk export operations
- `email_exporter: EmailExporter` - Email export functionality
- `format_customizer: FormatCustomizer` - Template customization
- `network_sharing: NetworkSharing` - Network sharing capabilities
- `pdf_generator: PDFGenerator` - PDF generation
- `usb_exporter: USBExporter` - USB export functionality
- `active_exports: dict` - Currently active export operations

**Methods:**
- `__init__(config: dict = None)` - Initialize export manager
- `export_session(session_id: str, format: str, method: str, options: dict)` - Export single session
- `export_sessions(session_ids: List[str], format: str, method: str, options: dict)` - Export multiple sessions
- `get_export_status(export_id: str)` - Get export operation status
- `cancel_export(export_id: str)` - Cancel export operation
- `get_available_formats()` - Get available export formats
- `get_available_methods()` - Get available export methods
- `set_callback(event_type: str, callback: callable)` - Set event callback

#### Functions

##### `export_session_to_file(session_id: str, format: str, file_path: str, options: dict = None)`
Export session to file with specified format.

##### `export_sessions_bulk(session_ids: List[str], export_config: dict)`
Export multiple sessions with bulk operations.

##### `send_session_via_email(session_id: str, recipient: str, subject: str, options: dict = None)`
Send session via email with attachments.

**Usage Example:**
```python
# Create export manager
export_manager = ExportManager({
    "temp_directory": "/tmp/exports",
    "max_concurrent_exports": 3,
    "enable_progress_tracking": True
})

# Export single session to PDF
export_id = export_manager.export_session(
    session_id="session_123",
    format="pdf",
    method="email",
    options={
        "recipient": "user@example.com",
        "subject": "Meeting Recording",
        "include_audio": True,
        "template": "professional"
    }
)

# Monitor export progress
def on_progress(export_id, progress):
    print(f"Export {export_id}: {progress:.1%} complete")

def on_complete(export_id, success, file_path):
    if success:
        print(f"Export completed: {file_path}")
    else:
        print(f"Export failed: {export_id}")

export_manager.set_callback("progress", on_progress)
export_manager.set_callback("complete", on_complete)

# Export multiple sessions
export_manager.export_sessions(
    session_ids=["session_1", "session_2", "session_3"],
    format="pdf",
    method="usb",
    options={
        "template": "summary",
        "combine_sessions": True
    }
)
```

### 2. `bulk_exporter.py`

**Purpose**: Multi-session batch export operations with job management and progress tracking.

#### Classes

##### `ExportJob`
Individual export job configuration.

**Attributes:**
- `job_id: str` - Job identifier
- `session_ids: List[str]` - Sessions to export
- `format: str` - Export format
- `method: str` - Export method
- `options: dict` - Export options
- `priority: int` - Job priority
- `created_at: datetime` - Job creation time
- `status: str` - Current job status

##### `BulkExportConfig`
Configuration for bulk export operations.

**Attributes:**
- `max_concurrent_jobs: int` - Maximum concurrent jobs
- `max_retries: int` - Maximum retry attempts
- `timeout: float` - Job timeout in seconds
- `chunk_size: int` - Sessions per chunk
- `progress_interval: float` - Progress update interval
- `temp_directory: str` - Temporary directory path
- `cleanup_temp_files: bool` - Clean up temporary files

##### `BulkExporter`
Main bulk export management system.

**Methods:**
- `__init__(config: BulkExportConfig)` - Initialize bulk exporter
- `add_export_job(job: ExportJob)` - Add job to queue
- `start_processing()` - Start job processing
- `stop_processing()` - Stop job processing
- `get_job_status(job_id: str)` - Get job status
- `cancel_job(job_id: str)` - Cancel job
- `get_queue_status()` - Get queue status
- `cleanup_completed_jobs()` - Clean up completed jobs
- `estimate_processing_time(job: ExportJob)` - Estimate processing time

**Usage Example:**
```python
# Create bulk export configuration
config = BulkExportConfig(
    max_concurrent_jobs=3,
    max_retries=3,
    timeout=300.0,
    chunk_size=5,
    progress_interval=1.0,
    temp_directory="/tmp/bulk_exports"
)

# Create bulk exporter
bulk_exporter = BulkExporter(config)

# Create export jobs
job1 = ExportJob(
    job_id="job_001",
    session_ids=["session_1", "session_2", "session_3"],
    format="pdf",
    method="email",
    options={
        "recipient": "manager@company.com",
        "subject": "Weekly Meeting Reports",
        "template": "summary"
    },
    priority=1
)

job2 = ExportJob(
    job_id="job_002",
    session_ids=["session_4", "session_5"],
    format="json",
    method="usb",
    options={
        "include_audio": False,
        "compress": True
    },
    priority=2
)

# Add jobs to queue
bulk_exporter.add_export_job(job1)
bulk_exporter.add_export_job(job2)

# Start processing
bulk_exporter.start_processing()

# Monitor progress
def on_job_progress(job_id, progress):
    print(f"Job {job_id}: {progress:.1%} complete")

def on_job_complete(job_id, success, results):
    if success:
        print(f"Job {job_id} completed successfully")
    else:
        print(f"Job {job_id} failed")

bulk_exporter.set_callback("job_progress", on_job_progress)
bulk_exporter.set_callback("job_complete", on_job_complete)

# Get queue status
status = bulk_exporter.get_queue_status()
print(f"Queue status: {status.pending_jobs} pending, {status.active_jobs} active")
```

### 3. `email_exporter.py`

**Purpose**: SMTP-based email delivery with attachment support and template customization.

#### Classes

##### `EmailConfig`
Email configuration settings.

**Attributes:**
- `smtp_server: str` - SMTP server address
- `smtp_port: int` - SMTP server port
- `username: str` - SMTP username
- `password: str` - SMTP password
- `use_tls: bool` - Use TLS encryption
- `use_ssl: bool` - Use SSL encryption
- `from_address: str` - From email address
- `from_name: str` - From display name
- `max_attachment_size: int` - Maximum attachment size in bytes
- `timeout: float` - Connection timeout

##### `EmailMessage`
Email message structure.

**Attributes:**
- `to_addresses: List[str]` - Recipient addresses
- `cc_addresses: List[str]` - CC addresses
- `bcc_addresses: List[str]` - BCC addresses
- `subject: str` - Email subject
- `body: str` - Email body
- `html_body: str` - HTML email body
- `attachments: List[dict]` - Email attachments
- `headers: dict` - Additional headers

##### `EmailExporter`
Main email export system.

**Methods:**
- `__init__(config: EmailConfig)` - Initialize email exporter
- `send_session_email(session_id: str, message: EmailMessage)` - Send session via email
- `send_bulk_email(session_ids: List[str], message: EmailMessage)` - Send multiple sessions
- `create_session_email(session_id: str, template: str, options: dict)` - Create session email
- `test_connection()` - Test SMTP connection
- `validate_email_address(address: str)` - Validate email address
- `get_delivery_status(message_id: str)` - Get delivery status

**Usage Example:**
```python
# Create email configuration
email_config = EmailConfig(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your.email@gmail.com",
    password="your_app_password",
    use_tls=True,
    from_address="meetings@company.com",
    from_name="Meeting Recorder",
    max_attachment_size=25 * 1024 * 1024  # 25MB
)

# Create email exporter
email_exporter = EmailExporter(email_config)

# Test connection
if email_exporter.test_connection():
    print("Email connection successful")

# Create email message
message = EmailMessage(
    to_addresses=["recipient@example.com"],
    subject="Meeting Recording - Team Standup",
    body="Please find the attached meeting recording and transcript.",
    attachments=[
        {
            "filename": "meeting_transcript.pdf",
            "content": pdf_content,
            "mime_type": "application/pdf"
        }
    ]
)

# Send session email
delivery_id = email_exporter.send_session_email("session_123", message)

# Monitor delivery status
status = email_exporter.get_delivery_status(delivery_id)
print(f"Delivery status: {status}")

# Create email from template
email_message = email_exporter.create_session_email(
    session_id="session_123",
    template="professional",
    options={
        "include_transcript": True,
        "include_audio": True,
        "include_summary": True
    }
)

# Send email
email_exporter.send_session_email("session_123", email_message)
```

### 4. `format_customizer.py`

**Purpose**: User-configurable export templates with variables, styling, and customization options.

#### Classes

##### `TemplateConfig`
Template configuration settings.

**Attributes:**
- `template_name: str` - Template name
- `description: str` - Template description
- `format: str` - Output format
- `variables: dict` - Template variables
- `styling: dict` - Style configuration
- `sections: List[str]` - Template sections
- `custom_css: str` - Custom CSS for HTML/PDF
- `custom_js: str` - Custom JavaScript for HTML

##### `TemplateVariable`
Template variable definition.

**Attributes:**
- `name: str` - Variable name
- `type: str` - Variable type
- `default_value: Any` - Default value
- `description: str` - Variable description
- `required: bool` - Is required
- `options: List[Any]` - Available options

##### `FormatCustomizer`
Main template customization system.

**Methods:**
- `__init__(config: dict = None)` - Initialize customizer
- `create_template(name: str, config: TemplateConfig)` - Create new template
- `update_template(name: str, config: TemplateConfig)` - Update template
- `delete_template(name: str)` - Delete template
- `get_template(name: str)` - Get template configuration
- `list_templates()` - List available templates
- `render_template(template_name: str, data: dict)` - Render template with data
- `validate_template(template_name: str)` - Validate template
- `import_template(file_path: str)` - Import template from file
- `export_template(template_name: str, file_path: str)` - Export template to file

**Usage Example:**
```python
# Create format customizer
customizer = FormatCustomizer()

# Create custom template
template_config = TemplateConfig(
    template_name="company_meeting",
    description="Company meeting report template",
    format="pdf",
    variables={
        "company_name": "ACME Corp",
        "logo_url": "https://company.com/logo.png",
        "report_date": "{{current_date}}",
        "include_action_items": True
    },
    styling={
        "font_family": "Arial",
        "font_size": 12,
        "header_color": "#003366",
        "primary_color": "#0066cc"
    },
    sections=["header", "summary", "transcript", "action_items", "footer"]
)

# Create template
customizer.create_template("company_meeting", template_config)

# Render template with session data
session_data = {
    "session_title": "Weekly Team Meeting",
    "date": "2024-01-15",
    "participants": ["Alice", "Bob", "Charlie"],
    "transcript": "Meeting transcript content...",
    "summary": "Meeting summary...",
    "action_items": ["Task 1", "Task 2", "Task 3"]
}

rendered_content = customizer.render_template("company_meeting", session_data)

# Save rendered content
with open("meeting_report.html", "w") as f:
    f.write(rendered_content)

# List available templates
templates = customizer.list_templates()
for template in templates:
    print(f"Template: {template.name} - {template.description}")

# Export template for sharing
customizer.export_template("company_meeting", "templates/company_meeting.json")
```

### 5. `network_sharing.py`

**Purpose**: SMB and HTTP file sharing capabilities for local network access.

#### Classes

##### `NetworkConfig`
Network sharing configuration.

**Attributes:**
- `enable_smb: bool` - Enable SMB sharing
- `enable_http: bool` - Enable HTTP sharing
- `smb_workgroup: str` - SMB workgroup
- `smb_server_name: str` - SMB server name
- `smb_share_name: str` - SMB share name
- `smb_username: str` - SMB username
- `smb_password: str` - SMB password
- `http_port: int` - HTTP server port
- `http_interface: str` - HTTP interface to bind
- `enable_authentication: bool` - Enable authentication
- `allowed_ips: List[str]` - Allowed IP addresses

##### `SMBShare`
SMB sharing functionality.

**Methods:**
- `__init__(config: NetworkConfig)` - Initialize SMB share
- `start_smb_server()` - Start SMB server
- `stop_smb_server()` - Stop SMB server
- `share_file(file_path: str, share_name: str)` - Share file via SMB
- `unshare_file(share_name: str)` - Remove file from share
- `get_share_url(share_name: str)` - Get SMB share URL
- `list_shared_files()` - List shared files

##### `HTTPShare`
HTTP sharing functionality.

**Methods:**
- `__init__(config: NetworkConfig)` - Initialize HTTP share
- `start_http_server()` - Start HTTP server
- `stop_http_server()` - Stop HTTP server
- `share_file(file_path: str, url_path: str)` - Share file via HTTP
- `unshare_file(url_path: str)` - Remove file from HTTP share
- `get_share_url(url_path: str)` - Get HTTP share URL
- `set_access_control(url_path: str, permissions: dict)` - Set access control

##### `NetworkSharing`
Main network sharing system.

**Methods:**
- `__init__(config: NetworkConfig)` - Initialize network sharing
- `start_sharing()` - Start all sharing services
- `stop_sharing()` - Stop all sharing services
- `share_session(session_id: str, method: str, options: dict)` - Share session
- `unshare_session(session_id: str)` - Remove session from sharing
- `get_sharing_status()` - Get sharing status
- `get_network_info()` - Get network information

**Usage Example:**
```python
# Create network configuration
network_config = NetworkConfig(
    enable_smb=True,
    enable_http=True,
    smb_workgroup="WORKGROUP",
    smb_server_name="MEETING_RECORDER",
    smb_share_name="meetings",
    http_port=8080,
    http_interface="0.0.0.0",
    enable_authentication=True,
    allowed_ips=["192.168.1.0/24"]
)

# Create network sharing
network_sharing = NetworkSharing(network_config)

# Start sharing services
network_sharing.start_sharing()

# Share session via HTTP
share_info = network_sharing.share_session(
    session_id="session_123",
    method="http",
    options={
        "format": "pdf",
        "template": "summary",
        "access_control": {
            "password": "meeting123",
            "expires": "2024-01-20"
        }
    }
)

print(f"Session shared at: {share_info.url}")
print(f"Access code: {share_info.access_code}")

# Share via SMB
smb_share_info = network_sharing.share_session(
    session_id="session_123",
    method="smb",
    options={
        "format": "json",
        "share_name": "weekly_meeting"
    }
)

print(f"SMB share: {smb_share_info.smb_path}")

# Get sharing status
status = network_sharing.get_sharing_status()
print(f"HTTP server: {status.http_status}")
print(f"SMB server: {status.smb_status}")
print(f"Active shares: {status.active_shares}")
```

### 6. `pdf_generator.py`

**Purpose**: Professional PDF generation using ReportLab with templates and customization.

#### Classes

##### `PDFConfig`
PDF generation configuration.

**Attributes:**
- `page_size: str` - Page size ("A4", "Letter", etc.)
- `orientation: str` - Page orientation ("portrait", "landscape")
- `margin_top: float` - Top margin
- `margin_bottom: float` - Bottom margin
- `margin_left: float` - Left margin
- `margin_right: float` - Right margin
- `font_family: str` - Default font family
- `font_size: int` - Default font size
- `header_font_size: int` - Header font size
- `include_header: bool` - Include header
- `include_footer: bool` - Include footer
- `include_page_numbers: bool` - Include page numbers

##### `PDFSection`
PDF section definition.

**Attributes:**
- `section_type: str` - Section type
- `title: str` - Section title
- `content: str` - Section content
- `styling: dict` - Section styling
- `page_break: bool` - Insert page break after section

##### `PDFGenerator`
Main PDF generation system.

**Methods:**
- `__init__(config: PDFConfig)` - Initialize PDF generator
- `create_session_pdf(session_id: str, template: str, options: dict)` - Create session PDF
- `add_header(content: str, styling: dict)` - Add header
- `add_footer(content: str, styling: dict)` - Add footer
- `add_section(section: PDFSection)` - Add section
- `add_table(data: List[List[str]], headers: List[str])` - Add table
- `add_image(image_path: str, width: float, height: float)` - Add image
- `generate_pdf(output_path: str)` - Generate PDF file
- `get_pdf_info()` - Get PDF information

**Usage Example:**
```python
# Create PDF configuration
pdf_config = PDFConfig(
    page_size="A4",
    orientation="portrait",
    margin_top=72,
    margin_bottom=72,
    margin_left=72,
    margin_right=72,
    font_family="Helvetica",
    font_size=12,
    include_header=True,
    include_footer=True,
    include_page_numbers=True
)

# Create PDF generator
pdf_generator = PDFGenerator(pdf_config)

# Create session PDF
pdf_content = pdf_generator.create_session_pdf(
    session_id="session_123",
    template="professional",
    options={
        "include_transcript": True,
        "include_summary": True,
        "include_action_items": True,
        "include_participant_stats": True
    }
)

# Add custom sections
summary_section = PDFSection(
    section_type="summary",
    title="Meeting Summary",
    content="This was a productive meeting where we discussed...",
    styling={
        "font_size": 14,
        "font_weight": "bold",
        "color": "#333333"
    }
)

pdf_generator.add_section(summary_section)

# Add participant table
participant_data = [
    ["Name", "Speaking Time", "Word Count", "Engagement"],
    ["Alice", "15:30", "450", "High"],
    ["Bob", "12:45", "380", "Medium"],
    ["Charlie", "8:20", "250", "Medium"]
]

pdf_generator.add_table(
    data=participant_data,
    headers=["Name", "Speaking Time", "Word Count", "Engagement"]
)

# Generate PDF
pdf_generator.generate_pdf("meeting_report.pdf")

# Get PDF information
pdf_info = pdf_generator.get_pdf_info()
print(f"PDF generated: {pdf_info.page_count} pages, {pdf_info.file_size} bytes")
```

### 7. `usb_exporter.py`

**Purpose**: USB drive detection and file transfer with progress tracking.

#### Classes

##### `USBConfig`
USB export configuration.

**Attributes:**
- `auto_detect: bool` - Enable auto-detection
- `preferred_filesystem: str` - Preferred filesystem
- `create_folder: bool` - Create folder structure
- `folder_name: str` - Folder name pattern
- `verify_transfer: bool` - Verify file transfer
- `eject_after_transfer: bool` - Eject drive after transfer
- `transfer_timeout: float` - Transfer timeout

##### `USBDevice`
USB device information.

**Attributes:**
- `device_path: str` - Device path
- `mount_point: str` - Mount point
- `filesystem: str` - Filesystem type
- `total_space: int` - Total space in bytes
- `free_space: int` - Free space in bytes
- `device_name: str` - Device name
- `vendor: str` - Device vendor
- `model: str` - Device model

##### `USBExporter`
Main USB export system.

**Methods:**
- `__init__(config: USBConfig)` - Initialize USB exporter
- `detect_usb_devices()` - Detect connected USB devices
- `select_device(device_path: str)` - Select USB device
- `export_session_to_usb(session_id: str, device: USBDevice, options: dict)` - Export session to USB
- `get_transfer_progress(transfer_id: str)` - Get transfer progress
- `cancel_transfer(transfer_id: str)` - Cancel transfer
- `eject_device(device: USBDevice)` - Eject USB device
- `verify_transfer(transfer_id: str)` - Verify transfer integrity

**Usage Example:**
```python
# Create USB configuration
usb_config = USBConfig(
    auto_detect=True,
    preferred_filesystem="FAT32",
    create_folder=True,
    folder_name="SilentSteno_{{date}}",
    verify_transfer=True,
    eject_after_transfer=True,
    transfer_timeout=300.0
)

# Create USB exporter
usb_exporter = USBExporter(usb_config)

# Detect USB devices
devices = usb_exporter.detect_usb_devices()
if devices:
    print(f"Found {len(devices)} USB devices:")
    for device in devices:
        print(f"  {device.device_name} - {device.free_space / 1024 / 1024:.1f} MB free")
    
    # Select first device
    selected_device = devices[0]
    usb_exporter.select_device(selected_device.device_path)
    
    # Export session to USB
    transfer_id = usb_exporter.export_session_to_usb(
        session_id="session_123",
        device=selected_device,
        options={
            "format": "pdf",
            "include_audio": True,
            "template": "summary",
            "folder_structure": True
        }
    )
    
    # Monitor transfer progress
    def on_progress(transfer_id, progress):
        print(f"Transfer {transfer_id}: {progress:.1%}")
    
    def on_complete(transfer_id, success):
        if success:
            print(f"Transfer {transfer_id} completed successfully")
            # Verify transfer
            verification = usb_exporter.verify_transfer(transfer_id)
            print(f"Verification: {verification.status}")
        else:
            print(f"Transfer {transfer_id} failed")
    
    usb_exporter.set_callback("progress", on_progress)
    usb_exporter.set_callback("complete", on_complete)
    
    # Wait for completion or cancel if needed
    # usb_exporter.cancel_transfer(transfer_id)
    
else:
    print("No USB devices detected")
```

## Module Integration

The Export module integrates with other Silent Steno components:

1. **Data Module**: Retrieves session data and metadata
2. **Core Events**: Publishes export events and progress updates
3. **AI Module**: Exports analysis results and insights
4. **Recording Module**: Exports audio files and transcripts
5. **UI Module**: Provides export dialogs and progress displays

## Common Usage Patterns

### Complete Export Workflow
```python
# Initialize export system
export_config = {
    "temp_directory": "/tmp/exports",
    "max_concurrent_exports": 3,
    "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "user@gmail.com",
        "password": "app_password"
    },
    "pdf": {
        "page_size": "A4",
        "font_family": "Arial",
        "include_header": True
    },
    "network": {
        "enable_http": True,
        "http_port": 8080
    }
}

export_manager = ExportManager(export_config)

# Create custom template
template_config = TemplateConfig(
    template_name="meeting_report",
    format="pdf",
    variables={
        "company_name": "ACME Corp",
        "include_summary": True,
        "include_action_items": True
    },
    styling={
        "primary_color": "#0066cc",
        "font_size": 12
    }
)

export_manager.format_customizer.create_template("meeting_report", template_config)

# Export session with multiple methods
session_id = "session_123"

# 1. Export to PDF and email
email_export = export_manager.export_session(
    session_id=session_id,
    format="pdf",
    method="email",
    options={
        "template": "meeting_report",
        "recipient": "manager@company.com",
        "subject": "Meeting Report",
        "include_audio": True
    }
)

# 2. Export to USB drive
usb_export = export_manager.export_session(
    session_id=session_id,
    format="json",
    method="usb",
    options={
        "include_transcript": True,
        "include_analysis": True,
        "verify_transfer": True
    }
)

# 3. Share via network
network_export = export_manager.export_session(
    session_id=session_id,
    format="html",
    method="network",
    options={
        "share_method": "http",
        "access_control": {"password": "meeting123"},
        "expires": "2024-01-20"
    }
)

# Monitor all exports
def monitor_exports():
    exports = [email_export, usb_export, network_export]
    for export_id in exports:
        status = export_manager.get_export_status(export_id)
        print(f"Export {export_id}: {status.status} - {status.progress:.1%}")

# Run monitoring
monitor_exports()
```

### Bulk Export Operations
```python
# Setup bulk export
bulk_config = BulkExportConfig(
    max_concurrent_jobs=2,
    max_retries=3,
    timeout=600.0,
    chunk_size=10
)

bulk_exporter = BulkExporter(bulk_config)

# Create bulk export jobs
sessions_by_date = {
    "2024-01-01": ["session_1", "session_2", "session_3"],
    "2024-01-02": ["session_4", "session_5"],
    "2024-01-03": ["session_6", "session_7", "session_8"]
}

jobs = []
for date, session_ids in sessions_by_date.items():
    job = ExportJob(
        job_id=f"export_{date}",
        session_ids=session_ids,
        format="pdf",
        method="email",
        options={
            "template": "daily_summary",
            "recipient": "archive@company.com",
            "subject": f"Daily Meeting Summary - {date}",
            "combine_sessions": True
        }
    )
    jobs.append(job)
    bulk_exporter.add_export_job(job)

# Start bulk processing
bulk_exporter.start_processing()

# Monitor bulk progress
def on_bulk_progress(job_id, progress):
    print(f"Bulk job {job_id}: {progress:.1%}")

def on_bulk_complete(job_id, success, results):
    if success:
        print(f"Bulk job {job_id} completed: {len(results)} files created")
    else:
        print(f"Bulk job {job_id} failed")

bulk_exporter.set_callback("job_progress", on_bulk_progress)
bulk_exporter.set_callback("job_complete", on_bulk_complete)
```

### Template Customization
```python
# Create custom templates for different purposes
templates = {
    "executive_summary": {
        "sections": ["header", "summary", "key_decisions", "action_items"],
        "styling": {
            "font_family": "Times New Roman",
            "primary_color": "#1a1a1a",
            "accent_color": "#0066cc"
        }
    },
    "detailed_report": {
        "sections": ["header", "participants", "full_transcript", "analysis", "appendix"],
        "styling": {
            "font_family": "Arial",
            "font_size": 10,
            "line_spacing": 1.2
        }
    },
    "action_items_only": {
        "sections": ["header", "action_items", "next_steps"],
        "styling": {
            "font_family": "Helvetica",
            "highlight_color": "#ffff00"
        }
    }
}

# Create all templates
for template_name, config in templates.items():
    template_config = TemplateConfig(
        template_name=template_name,
        format="pdf",
        sections=config["sections"],
        styling=config["styling"]
    )
    export_manager.format_customizer.create_template(template_name, template_config)

# Export with different templates
session_ids = ["session_1", "session_2", "session_3"]

# Executive summary for executives
export_manager.export_sessions(
    session_ids=session_ids,
    format="pdf",
    method="email",
    options={
        "template": "executive_summary",
        "recipient": "executives@company.com",
        "subject": "Executive Summary - Team Meetings"
    }
)

# Detailed report for team leads
export_manager.export_sessions(
    session_ids=session_ids,
    format="pdf",
    method="usb",
    options={
        "template": "detailed_report",
        "combine_sessions": False
    }
)

# Action items for project managers
export_manager.export_sessions(
    session_ids=session_ids,
    format="html",
    method="network",
    options={
        "template": "action_items_only",
        "share_method": "http"
    }
)
```

### Network Sharing Setup
```python
# Configure network sharing
network_config = NetworkConfig(
    enable_http=True,
    enable_smb=True,
    http_port=8080,
    smb_workgroup="OFFICE",
    smb_share_name="meetings",
    enable_authentication=True
)

network_sharing = NetworkSharing(network_config)
network_sharing.start_sharing()

# Share sessions with different access levels
# Public access for general meetings
public_share = network_sharing.share_session(
    session_id="general_meeting",
    method="http",
    options={
        "format": "html",
        "template": "summary",
        "access_control": {"public": True}
    }
)

# Password protected for sensitive meetings
secure_share = network_sharing.share_session(
    session_id="confidential_meeting",
    method="http",
    options={
        "format": "pdf",
        "template": "detailed_report",
        "access_control": {
            "password": "secure123",
            "expires": "2024-01-31",
            "max_downloads": 5
        }
    }
)

# SMB share for team access
team_share = network_sharing.share_session(
    session_id="team_meeting",
    method="smb",
    options={
        "format": "json",
        "share_name": "team_meetings",
        "access_control": {"group": "team_members"}
    }
)

print(f"Public share: {public_share.url}")
print(f"Secure share: {secure_share.url}")
print(f"SMB share: {team_share.smb_path}")
```

### Error Handling and Recovery
```python
# Robust export with error handling
class ExportHandler:
    def __init__(self, export_manager):
        self.export_manager = export_manager
        self.failed_exports = []
        self.retry_count = {}
        self.max_retries = 3
    
    def export_with_retry(self, session_id, format, method, options):
        """Export with automatic retry on failure."""
        export_id = f"{session_id}_{format}_{method}"
        
        for attempt in range(self.max_retries):
            try:
                result = self.export_manager.export_session(
                    session_id=session_id,
                    format=format,
                    method=method,
                    options=options
                )
                
                # Monitor export completion
                while True:
                    status = self.export_manager.get_export_status(result)
                    if status.status == "completed":
                        return result
                    elif status.status == "failed":
                        raise Exception(f"Export failed: {status.error}")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Export attempt {attempt + 1} failed: {e}")
                self.retry_count[export_id] = attempt + 1
                
                if attempt < self.max_retries - 1:
                    # Wait before retry
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Max retries reached
                    self.failed_exports.append({
                        "session_id": session_id,
                        "format": format,
                        "method": method,
                        "error": str(e)
                    })
                    
        return None
    
    def export_with_fallback(self, session_id, primary_method, fallback_method, options):
        """Export with fallback method on failure."""
        # Try primary method
        result = self.export_with_retry(
            session_id, options["format"], primary_method, options
        )
        
        if result:
            return result
        
        # Try fallback method
        print(f"Primary method failed, trying fallback: {fallback_method}")
        fallback_options = options.copy()
        fallback_options["method"] = fallback_method
        
        return self.export_with_retry(
            session_id, options["format"], fallback_method, fallback_options
        )
    
    def get_failed_exports(self):
        """Get list of failed exports."""
        return self.failed_exports

# Use robust export handler
export_handler = ExportHandler(export_manager)

# Export with fallback
result = export_handler.export_with_fallback(
    session_id="session_123",
    primary_method="email",
    fallback_method="usb",
    options={
        "format": "pdf",
        "template": "summary",
        "recipient": "user@example.com"
    }
)

if result:
    print(f"Export successful: {result}")
else:
    print("Export failed with all methods")
    failed = export_handler.get_failed_exports()
    for failure in failed:
        print(f"Failed: {failure}")
```

This comprehensive documentation provides complete technical details and practical usage examples for all components in the Export module, enabling developers to implement robust export and sharing capabilities for The Silent Steno system.