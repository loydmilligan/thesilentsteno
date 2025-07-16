"""
Email Export System

SMTP-based email delivery with PDF attachments and customizable templates.
"""

import smtplib
import ssl
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formatdate
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    """Email configuration settings"""
    smtp_server: str
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    sender_email: str = ""
    sender_name: str = "Silent Steno Device"
    reply_to: Optional[str] = None
    timeout: int = 30

@dataclass
class AttachmentConfig:
    """Email attachment configuration"""
    max_size_mb: int = 25
    allowed_types: List[str] = field(default_factory=lambda: ['.pdf', '.txt', '.json', '.html', '.zip'])
    compress_large_files: bool = True
    split_large_files: bool = False

@dataclass
class EmailTemplate:
    """Email template configuration"""
    subject_template: str = "Meeting Recording - {session_title} ({date})"
    body_template: str = """
Dear Recipient,

Please find attached the meeting recording and analysis for:

Session: {session_title}
Date: {date}
Duration: {duration}
Participants: {participant_count}

Summary:
{summary}

Best regards,
Silent Steno Device
"""
    html_template: Optional[str] = None
    use_html: bool = False

class EmailExporter:
    """SMTP-based email export system"""
    
    def __init__(self, config: EmailConfig):
        self.config = config
        self.attachment_config = AttachmentConfig()
        self.template = EmailTemplate()
        
    def send_session_email(self, session_id: str, file_path: str, 
                          recipients: List[str], **kwargs) -> bool:
        """Send session data via email"""
        try:
            # Load session data
            session_data = self._load_session_data(session_id)
            
            # Create email message
            msg = self._create_email_message(session_data, recipients, **kwargs)
            
            # Add attachments
            if file_path and os.path.exists(file_path):
                self._add_attachment(msg, file_path)
            
            # Send email
            success = self._send_email(msg, recipients)
            
            if success:
                logger.info(f"Email sent successfully for session {session_id} to {len(recipients)} recipients")
            else:
                logger.error(f"Failed to send email for session {session_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Email export failed for session {session_id}: {str(e)}")
            return False
    
    def _load_session_data(self, session_id: str) -> Dict[str, Any]:
        """Load session data from database"""
        # This would integrate with the database system
        # For now, return mock data
        return {
            'session_id': session_id,
            'title': f'Meeting Session {session_id}',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'duration': '45 minutes',
            'participant_count': 3,
            'summary': 'Important meeting discussing project milestones and next steps.'
        }
    
    def _create_email_message(self, session_data: Dict[str, Any], 
                            recipients: List[str], **kwargs) -> MIMEMultipart:
        """Create email message with template"""
        msg = MIMEMultipart('alternative')
        
        # Format subject
        subject = self.template.subject_template.format(**session_data)
        msg['Subject'] = subject
        msg['From'] = f"{self.config.sender_name} <{self.config.sender_email}>"
        msg['To'] = ', '.join(recipients)
        msg['Date'] = formatdate(localtime=True)
        
        if self.config.reply_to:
            msg['Reply-To'] = self.config.reply_to
        
        # Format body
        body_text = self.template.body_template.format(**session_data)
        text_part = MIMEText(body_text, 'plain')
        msg.attach(text_part)
        
        # Add HTML part if configured
        if self.template.use_html and self.template.html_template:
            body_html = self.template.html_template.format(**session_data)
            html_part = MIMEText(body_html, 'html')
            msg.attach(html_part)
        
        return msg
    
    def _add_attachment(self, msg: MIMEMultipart, file_path: str):
        """Add file attachment to email"""
        try:
            file_path = Path(file_path)
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.attachment_config.max_size_mb:
                logger.warning(f"File {file_path} exceeds size limit ({file_size_mb:.1f}MB)")
                if not self.attachment_config.compress_large_files:
                    return False
            
            # Check file type
            if file_path.suffix.lower() not in self.attachment_config.allowed_types:
                logger.warning(f"File type {file_path.suffix} not allowed for email")
                return False
            
            # Add attachment
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {file_path.name}'
            )
            
            msg.attach(part)
            logger.info(f"Added attachment: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add attachment {file_path}: {str(e)}")
            return False
    
    def _send_email(self, msg: MIMEMultipart, recipients: List[str]) -> bool:
        """Send email via SMTP"""
        try:
            # Create SMTP connection
            if self.config.use_ssl:
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(self.config.smtp_server, 
                                        self.config.smtp_port, 
                                        context=context, 
                                        timeout=self.config.timeout)
            else:
                server = smtplib.SMTP(self.config.smtp_server, 
                                    self.config.smtp_port, 
                                    timeout=self.config.timeout)
                
                if self.config.use_tls:
                    context = ssl.create_default_context()
                    server.starttls(context=context)
            
            # Login if credentials provided
            if self.config.username and self.config.password:
                server.login(self.config.username, self.config.password)
            
            # Send email
            text = msg.as_string()
            server.sendmail(self.config.sender_email, recipients, text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"SMTP send failed: {str(e)}")
            return False
    
    def validate_email_config(self) -> bool:
        """Validate email configuration"""
        try:
            # Check required fields
            if not self.config.smtp_server:
                logger.error("SMTP server not configured")
                return False
            
            if not self.config.sender_email:
                logger.error("Sender email not configured")
                return False
            
            # Validate email format
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, self.config.sender_email):
                logger.error("Invalid sender email format")
                return False
            
            # Test SMTP connection
            try:
                if self.config.use_ssl:
                    context = ssl.create_default_context()
                    server = smtplib.SMTP_SSL(self.config.smtp_server, 
                                            self.config.smtp_port, 
                                            context=context,
                                            timeout=10)
                else:
                    server = smtplib.SMTP(self.config.smtp_server, 
                                        self.config.smtp_port,
                                        timeout=10)
                    
                    if self.config.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                
                server.quit()
                logger.info("SMTP connection test successful")
                return True
                
            except Exception as e:
                logger.error(f"SMTP connection test failed: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Email configuration validation failed: {str(e)}")
            return False
    
    def update_template(self, template: EmailTemplate):
        """Update email template"""
        self.template = template
        logger.info("Email template updated")
    
    def update_attachment_config(self, config: AttachmentConfig):
        """Update attachment configuration"""
        self.attachment_config = config
        logger.info("Attachment configuration updated")

def create_email_exporter(config: Optional[EmailConfig] = None) -> EmailExporter:
    """Create email exporter with default or provided configuration"""
    if config is None:
        # Default configuration - should be loaded from settings
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            use_tls=True,
            sender_email="silentSteno@example.com",
            sender_name="Silent Steno Device"
        )
    
    return EmailExporter(config)

def send_session_email(session_id: str, file_path: str, recipients: List[str], 
                      config: Optional[EmailConfig] = None) -> bool:
    """Convenience function to send session email"""
    exporter = create_email_exporter(config)
    return exporter.send_session_email(session_id, file_path, recipients)

def validate_email_config(email: str) -> bool:
    """Validate email address format"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def create_email_template(subject: str, body: str, html_body: Optional[str] = None) -> EmailTemplate:
    """Create email template"""
    return EmailTemplate(
        subject_template=subject,
        body_template=body,
        html_template=html_body,
        use_html=html_body is not None
    )

def format_email_content(template: str, session_data: Dict[str, Any]) -> str:
    """Format email content with session data"""
    return template.format(**session_data)