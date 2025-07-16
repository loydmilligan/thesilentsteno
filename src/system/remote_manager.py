"""
Remote Device Management

Remote management capabilities for The Silent Steno device including
secure remote access, command execution, and monitoring.
"""

import os
import json
import logging
import threading
import time
import ssl
import socket
import hashlib
import secrets
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import jwt
from urllib.parse import urlparse, parse_qs


class RemoteAccessLevel(Enum):
    READ_ONLY = "read_only"
    MONITOR = "monitor"
    CONTROL = "control"
    ADMIN = "admin"


class RemoteSessionStatus(Enum):
    INACTIVE = "inactive"
    CONNECTING = "connecting"
    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    EXPIRED = "expired"


class RemoteCommandType(Enum):
    STATUS = "status"
    LOGS = "logs"
    RESTART = "restart"
    UPDATE = "update"
    BACKUP = "backup"
    DIAGNOSTICS = "diagnostics"
    FACTORY_RESET = "factory_reset"
    CONFIGURATION = "configuration"


@dataclass
class RemoteCredentials:
    """Remote access credentials."""
    username: str
    access_level: RemoteAccessLevel
    api_key: str
    session_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    allowed_commands: List[RemoteCommandType] = field(default_factory=list)
    ip_restrictions: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if credentials are expired."""
        if not self.expires_at:
            return False
        return datetime.now() >= self.expires_at
    
    def can_execute_command(self, command: RemoteCommandType) -> bool:
        """Check if credentials allow executing a command."""
        if self.is_expired():
            return False
        
        if not self.allowed_commands:
            # Default permissions based on access level
            if self.access_level == RemoteAccessLevel.READ_ONLY:
                return command in [RemoteCommandType.STATUS, RemoteCommandType.LOGS]
            elif self.access_level == RemoteAccessLevel.MONITOR:
                return command in [RemoteCommandType.STATUS, RemoteCommandType.LOGS, RemoteCommandType.DIAGNOSTICS]
            elif self.access_level == RemoteAccessLevel.CONTROL:
                return command != RemoteCommandType.FACTORY_RESET
            elif self.access_level == RemoteAccessLevel.ADMIN:
                return True
        
        return command in self.allowed_commands


@dataclass
class RemoteSession:
    """Remote management session."""
    session_id: str
    credentials: RemoteCredentials
    client_ip: str
    status: RemoteSessionStatus
    started_at: datetime
    last_activity: datetime
    commands_executed: int = 0
    errors: List[str] = field(default_factory=list)
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == RemoteSessionStatus.ACTIVE and not self.credentials.is_expired()
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


@dataclass
class RemoteCommand:
    """Remote command execution request."""
    command_type: RemoteCommandType
    parameters: Dict[str, Any]
    session_id: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "command_type": self.command_type.value,
            "parameters": self.parameters,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RemoteCommandResult:
    """Result of remote command execution."""
    success: bool
    command_type: RemoteCommandType
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "command_type": self.command_type.value,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


class RemoteCommandExecutor:
    """Executes remote commands securely."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.command_handlers = {
            RemoteCommandType.STATUS: self._handle_status,
            RemoteCommandType.LOGS: self._handle_logs,
            RemoteCommandType.RESTART: self._handle_restart,
            RemoteCommandType.UPDATE: self._handle_update,
            RemoteCommandType.BACKUP: self._handle_backup,
            RemoteCommandType.DIAGNOSTICS: self._handle_diagnostics,
            RemoteCommandType.FACTORY_RESET: self._handle_factory_reset,
            RemoteCommandType.CONFIGURATION: self._handle_configuration
        }
    
    def execute_command(self, command: RemoteCommand, session: RemoteSession) -> RemoteCommandResult:
        """Execute a remote command."""
        start_time = time.time()
        
        try:
            # Check permissions
            if not session.credentials.can_execute_command(command.command_type):
                return RemoteCommandResult(
                    success=False,
                    command_type=command.command_type,
                    result_data={},
                    error_message="Access denied for this command"
                )
            
            # Get command handler
            handler = self.command_handlers.get(command.command_type)
            if not handler:
                return RemoteCommandResult(
                    success=False,
                    command_type=command.command_type,
                    result_data={},
                    error_message=f"Unknown command type: {command.command_type.value}"
                )
            
            # Execute command
            result_data = handler(command.parameters)
            
            return RemoteCommandResult(
                success=True,
                command_type=command.command_type,
                result_data=result_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error executing remote command {command.command_type.value}: {e}")
            return RemoteCommandResult(
                success=False,
                command_type=command.command_type,
                result_data={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _handle_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status command."""
        # Mock implementation - would integrate with actual system status
        return {
            "system_status": "running",
            "uptime": "2 days, 3 hours",
            "cpu_usage": 45.2,
            "memory_usage": 62.1,
            "disk_usage": 78.5,
            "bluetooth_connected": True,
            "audio_recording": False,
            "last_session": "2024-01-15T10:30:00Z"
        }
    
    def _handle_logs(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle logs command."""
        lines = parameters.get("lines", 100)
        log_level = parameters.get("level", "INFO")
        
        # Mock implementation - would read actual logs
        return {
            "logs": [
                {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "message": "System started"},
                {"timestamp": "2024-01-15T10:31:00Z", "level": "DEBUG", "message": "Audio system initialized"},
                {"timestamp": "2024-01-15T10:32:00Z", "level": "INFO", "message": "Bluetooth connected"}
            ],
            "lines_returned": 3,
            "log_level": log_level
        }
    
    def _handle_restart(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle restart command."""
        restart_type = parameters.get("type", "soft")
        
        # Mock implementation - would perform actual restart
        return {
            "restart_type": restart_type,
            "scheduled_at": datetime.now().isoformat(),
            "estimated_downtime": "30 seconds"
        }
    
    def _handle_update(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update command."""
        update_type = parameters.get("type", "check")
        
        # Mock implementation - would integrate with update manager
        if update_type == "check":
            return {
                "updates_available": True,
                "latest_version": "0.1.1",
                "current_version": "0.1.0",
                "update_size": "5.2 MB",
                "release_notes": "Bug fixes and improvements"
            }
        else:
            return {
                "update_started": True,
                "estimated_time": "5 minutes",
                "restart_required": True
            }
    
    def _handle_backup(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle backup command."""
        backup_type = parameters.get("type", "full")
        
        # Mock implementation - would integrate with backup system
        return {
            "backup_type": backup_type,
            "backup_started": True,
            "estimated_size": "150 MB",
            "estimated_time": "2 minutes"
        }
    
    def _handle_diagnostics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle diagnostics command."""
        test_type = parameters.get("type", "full")
        
        # Mock implementation - would integrate with diagnostic system
        return {
            "test_type": test_type,
            "tests_passed": 15,
            "tests_failed": 2,
            "overall_health": "Good",
            "critical_issues": 0,
            "warnings": 3
        }
    
    def _handle_factory_reset(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle factory reset command."""
        confirmation = parameters.get("confirm", False)
        
        if not confirmation:
            return {
                "confirmation_required": True,
                "warning": "This will erase all data and restore factory defaults"
            }
        
        # Mock implementation - would perform actual factory reset
        return {
            "factory_reset_started": True,
            "estimated_time": "10 minutes",
            "backup_created": True
        }
    
    def _handle_configuration(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration command."""
        action = parameters.get("action", "get")
        
        if action == "get":
            return {
                "configuration": {
                    "device_name": "Silent Steno Device",
                    "audio_quality": "high",
                    "auto_cleanup": True,
                    "bluetooth_discoverable": False
                }
            }
        else:
            return {
                "configuration_updated": True,
                "restart_required": False
            }


class RemoteAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for remote API requests."""
    
    def __init__(self, *args, remote_manager=None, **kwargs):
        self.remote_manager = remote_manager
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            if self.path.startswith('/api/status'):
                self._handle_status_request()
            elif self.path.startswith('/api/logs'):
                self._handle_logs_request()
            else:
                self._send_error(404, "Not Found")
        except Exception as e:
            self._send_error(500, str(e))
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            if self.path.startswith('/api/login'):
                self._handle_login_request()
            elif self.path.startswith('/api/command'):
                self._handle_command_request()
            else:
                self._send_error(404, "Not Found")
        except Exception as e:
            self._send_error(500, str(e))
    
    def _handle_status_request(self):
        """Handle status request."""
        session = self._authenticate_request()
        if not session:
            return
        
        command = RemoteCommand(
            command_type=RemoteCommandType.STATUS,
            parameters={},
            session_id=session.session_id,
            timestamp=datetime.now()
        )
        
        result = self.remote_manager.executor.execute_command(command, session)
        self._send_json_response(result.to_dict())
    
    def _handle_logs_request(self):
        """Handle logs request."""
        session = self._authenticate_request()
        if not session:
            return
        
        # Parse query parameters
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        command = RemoteCommand(
            command_type=RemoteCommandType.LOGS,
            parameters={
                "lines": int(query_params.get("lines", [100])[0]),
                "level": query_params.get("level", ["INFO"])[0]
            },
            session_id=session.session_id,
            timestamp=datetime.now()
        )
        
        result = self.remote_manager.executor.execute_command(command, session)
        self._send_json_response(result.to_dict())
    
    def _handle_login_request(self):
        """Handle login request."""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            username = data.get('username')
            api_key = data.get('api_key')
            
            if not username or not api_key:
                self._send_error(400, "Missing username or api_key")
                return
            
            session = self.remote_manager.authenticate(username, api_key, self.client_address[0])
            if session:
                self._send_json_response({
                    "success": True,
                    "session_id": session.session_id,
                    "expires_at": session.credentials.expires_at.isoformat() if session.credentials.expires_at else None
                })
            else:
                self._send_error(401, "Authentication failed")
                
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
    
    def _handle_command_request(self):
        """Handle command execution request."""
        session = self._authenticate_request()
        if not session:
            return
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            command_type = RemoteCommandType(data.get('command_type'))
            parameters = data.get('parameters', {})
            
            command = RemoteCommand(
                command_type=command_type,
                parameters=parameters,
                session_id=session.session_id,
                timestamp=datetime.now()
            )
            
            result = self.remote_manager.executor.execute_command(command, session)
            self._send_json_response(result.to_dict())
            
        except (json.JSONDecodeError, ValueError) as e:
            self._send_error(400, f"Invalid request: {e}")
    
    def _authenticate_request(self) -> Optional[RemoteSession]:
        """Authenticate request and return session."""
        auth_header = self.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            self._send_error(401, "Missing or invalid authorization header")
            return None
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        session = self.remote_manager.get_session_by_token(token)
        
        if not session or not session.is_active():
            self._send_error(401, "Invalid or expired session")
            return None
        
        session.update_activity()
        return session
    
    def _send_json_response(self, data: Dict[str, Any]):
        """Send JSON response."""
        response = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        error_response = json.dumps({"error": message}).encode('utf-8')
        self.wfile.write(error_response)
    
    def log_message(self, format, *args):
        """Override to suppress default logging."""
        pass


class RemoteManager:
    """Main remote management system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.executor = RemoteCommandExecutor()
        self.sessions: Dict[str, RemoteSession] = {}
        self.credentials: Dict[str, RemoteCredentials] = {}
        self.server: Optional[HTTPServer] = None
        self.running = False
        self._lock = threading.RLock()
        
        # Load configuration
        self.port = self.config.get("port", 8443)
        self.host = self.config.get("host", "0.0.0.0")
        self.ssl_enabled = self.config.get("ssl_enabled", True)
        self.session_timeout = self.config.get("session_timeout", 3600)  # 1 hour
        
        # Load credentials
        self._load_credentials()
        
        # Setup cleanup thread
        self.cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self):
        """Start remote management server."""
        if self.running:
            return
        
        try:
            # Create server
            handler = lambda *args, **kwargs: RemoteAPIHandler(*args, remote_manager=self, **kwargs)
            self.server = HTTPServer((self.host, self.port), handler)
            
            # Setup SSL if enabled
            if self.ssl_enabled:
                self._setup_ssl()
            
            # Start server in separate thread
            server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            server_thread.start()
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            
            self.running = True
            self.logger.info(f"Remote management server started on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start remote management server: {e}")
            raise
    
    def stop(self):
        """Stop remote management server."""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        self.logger.info("Remote management server stopped")
    
    def authenticate(self, username: str, api_key: str, client_ip: str) -> Optional[RemoteSession]:
        """Authenticate user and create session."""
        try:
            with self._lock:
                credentials = self.credentials.get(username)
                if not credentials or credentials.api_key != api_key:
                    self.logger.warning(f"Authentication failed for user {username} from {client_ip}")
                    return None
                
                # Check IP restrictions
                if credentials.ip_restrictions and client_ip not in credentials.ip_restrictions:
                    self.logger.warning(f"IP {client_ip} not allowed for user {username}")
                    return None
                
                # Create session
                session_id = secrets.token_urlsafe(32)
                session_token = secrets.token_urlsafe(32)
                
                # Update credentials with session info
                credentials.session_token = session_token
                credentials.expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
                
                session = RemoteSession(
                    session_id=session_id,
                    credentials=credentials,
                    client_ip=client_ip,
                    status=RemoteSessionStatus.ACTIVE,
                    started_at=datetime.now(),
                    last_activity=datetime.now()
                )
                
                self.sessions[session_token] = session
                self.logger.info(f"User {username} authenticated from {client_ip}")
                return session
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None
    
    def get_session_by_token(self, token: str) -> Optional[RemoteSession]:
        """Get session by token."""
        with self._lock:
            return self.sessions.get(token)
    
    def _load_credentials(self):
        """Load user credentials from configuration."""
        # Default admin credentials
        self.credentials = {
            "admin": RemoteCredentials(
                username="admin",
                access_level=RemoteAccessLevel.ADMIN,
                api_key="admin_key_123",  # In production, this would be secure
                allowed_commands=list(RemoteCommandType)
            ),
            "monitor": RemoteCredentials(
                username="monitor",
                access_level=RemoteAccessLevel.MONITOR,
                api_key="monitor_key_456",
                allowed_commands=[RemoteCommandType.STATUS, RemoteCommandType.LOGS, RemoteCommandType.DIAGNOSTICS]
            )
        }
    
    def _setup_ssl(self):
        """Setup SSL for secure communication."""
        try:
            # In production, use proper SSL certificates
            import ssl
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            # context.load_cert_chain(certfile, keyfile)
            self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
        except Exception as e:
            self.logger.warning(f"SSL setup failed: {e}")
    
    def _cleanup_loop(self):
        """Cleanup expired sessions."""
        while self.running and not self._stop_event.is_set():
            try:
                with self._lock:
                    expired_tokens = []
                    for token, session in self.sessions.items():
                        if not session.is_active():
                            expired_tokens.append(token)
                    
                    for token in expired_tokens:
                        del self.sessions[token]
                        self.logger.debug(f"Cleaned up expired session: {token}")
                
                # Wait before next cleanup
                if self._stop_event.wait(300):  # 5 minutes
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)
    
    def get_management_status(self) -> Dict[str, Any]:
        """Get remote management status."""
        with self._lock:
            return {
                "server_running": self.running,
                "host": self.host,
                "port": self.port,
                "ssl_enabled": self.ssl_enabled,
                "active_sessions": len(self.sessions),
                "registered_users": len(self.credentials),
                "session_timeout": self.session_timeout
            }


# Factory functions
def create_remote_manager(config: Dict[str, Any] = None) -> RemoteManager:
    """Create a remote manager instance."""
    return RemoteManager(config)


def start_remote_management(manager: RemoteManager = None, config: Dict[str, Any] = None):
    """Start remote management server."""
    if manager is None:
        manager = create_remote_manager(config)
    
    manager.start()
    return manager


def stop_remote_management(manager: RemoteManager):
    """Stop remote management server."""
    manager.stop()


def execute_remote_command(command_type: RemoteCommandType, 
                          parameters: Dict[str, Any], 
                          session: RemoteSession,
                          manager: RemoteManager = None) -> RemoteCommandResult:
    """Execute a remote command."""
    if manager is None:
        manager = create_remote_manager()
    
    command = RemoteCommand(
        command_type=command_type,
        parameters=parameters,
        session_id=session.session_id,
        timestamp=datetime.now()
    )
    
    return manager.executor.execute_command(command, session)