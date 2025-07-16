"""
Network Sharing System

SMB and HTTP file sharing for local network access to meeting recordings.
"""

import os
import logging
import socket
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)

@dataclass
class SharingConfig:
    """Network sharing configuration"""
    smb_enabled: bool = True
    http_enabled: bool = True
    smb_port: int = 445
    http_port: int = 8080
    share_name: str = "SilentSteno"
    share_path: str = "/tmp/silentSteno_share"
    access_password: Optional[str] = None
    read_only: bool = True
    max_connections: int = 10
    session_timeout: int = 3600  # 1 hour
    allowed_ips: List[str] = field(default_factory=list)
    log_access: bool = True

class SMBServer:
    """SMB/CIFS file sharing server"""
    
    def __init__(self, config: SharingConfig):
        self.config = config
        self.running = False
        self.server_process: Optional[subprocess.Popen] = None
        self.share_config_file: Optional[str] = None
        
    def start(self) -> bool:
        """Start SMB server"""
        try:
            if self.running:
                logger.warning("SMB server already running")
                return True
            
            # Ensure share directory exists
            Path(self.config.share_path).mkdir(parents=True, exist_ok=True)
            
            # Create Samba configuration
            self._create_samba_config()
            
            # Start Samba daemon
            cmd = ['smbd', '--foreground', '--configfile', self.share_config_file]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment to check if process started successfully
            time.sleep(1)
            if self.server_process.poll() is None:
                self.running = True
                logger.info(f"SMB server started on port {self.config.smb_port}")
                return True
            else:
                logger.error("Failed to start SMB server")
                return False
                
        except Exception as e:
            logger.error(f"SMB server start failed: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """Stop SMB server"""
        try:
            if not self.running:
                return True
            
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            
            # Cleanup config file
            if self.share_config_file and os.path.exists(self.share_config_file):
                os.unlink(self.share_config_file)
            
            self.running = False
            logger.info("SMB server stopped")
            return True
            
        except Exception as e:
            logger.error(f"SMB server stop failed: {str(e)}")
            return False
    
    def _create_samba_config(self):
        """Create Samba configuration file"""
        config_content = f"""
[global]
    workgroup = WORKGROUP
    server string = Silent Steno Device
    netbios name = SILENTSTENODEVICE
    security = user
    guest account = nobody
    map to guest = bad user
    socket options = TCP_NODELAY SO_RCVBUF=8192 SO_SNDBUF=8192
    max connections = {self.config.max_connections}
    deadtime = {self.config.session_timeout // 60}
    log level = 1
    
[{self.config.share_name}]
    comment = Silent Steno Meeting Recordings
    path = {self.config.share_path}
    browseable = yes
    guest ok = yes
    read only = {'yes' if self.config.read_only else 'no'}
    create mask = 0644
    directory mask = 0755
"""
        
        # Add IP restrictions if configured
        if self.config.allowed_ips:
            hosts_allow = ' '.join(self.config.allowed_ips)
            config_content += f"    hosts allow = {hosts_allow}\n"
        
        # Create temporary config file
        fd, self.share_config_file = tempfile.mkstemp(suffix='.conf', prefix='smb_')
        with os.fdopen(fd, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Created SMB config: {self.share_config_file}")
    
    def add_file(self, file_path: str, share_filename: Optional[str] = None) -> bool:
        """Add file to SMB share"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                logger.error(f"Source file does not exist: {file_path}")
                return False
            
            dest_filename = share_filename or source_path.name
            dest_path = Path(self.config.share_path) / dest_filename
            
            # Copy file to share directory
            shutil.copy2(source_path, dest_path)
            
            logger.info(f"Added file to SMB share: {dest_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add file to SMB share: {str(e)}")
            return False
    
    def remove_file(self, filename: str) -> bool:
        """Remove file from SMB share"""
        try:
            file_path = Path(self.config.share_path) / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed file from SMB share: {filename}")
                return True
            else:
                logger.warning(f"File not found in SMB share: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove file from SMB share: {str(e)}")
            return False

class HTTPServer:
    """HTTP file sharing server"""
    
    def __init__(self, config: SharingConfig):
        self.config = config
        self.running = False
        self.server: Optional[socketserver.TCPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        
    def start(self) -> bool:
        """Start HTTP server"""
        try:
            if self.running:
                logger.warning("HTTP server already running")
                return True
            
            # Ensure share directory exists
            Path(self.config.share_path).mkdir(parents=True, exist_ok=True)
            
            # Change to share directory for serving files
            original_cwd = os.getcwd()
            os.chdir(self.config.share_path)
            
            # Create custom handler class
            handler_class = self._create_handler_class()
            
            # Create server
            self.server = socketserver.TCPServer(
                ("0.0.0.0", self.config.http_port),
                handler_class
            )
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.running = True
            logger.info(f"HTTP server started on port {self.config.http_port}")
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            return True
            
        except Exception as e:
            logger.error(f"HTTP server start failed: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """Stop HTTP server"""
        try:
            if not self.running:
                return True
            
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            
            if self.server_thread:
                self.server_thread.join(timeout=5)
            
            self.running = False
            logger.info("HTTP server stopped")
            return True
            
        except Exception as e:
            logger.error(f"HTTP server stop failed: {str(e)}")
            return False
    
    def _create_handler_class(self):
        """Create custom HTTP handler class"""
        config = self.config
        
        class CustomHTTPHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                if config.log_access:
                    logger.info(f"HTTP access: {self.client_address[0]} - {format % args}")
            
            def do_GET(self):
                # Check IP restrictions
                if config.allowed_ips and self.client_address[0] not in config.allowed_ips:
                    self.send_error(403, "Access denied")
                    return
                
                # Check password if configured
                if config.access_password:
                    auth_header = self.headers.get('Authorization')
                    if not auth_header or not self._check_auth(auth_header):
                        self.send_response(401)
                        self.send_header('WWW-Authenticate', 'Basic realm="Silent Steno"')
                        self.end_headers()
                        return
                
                super().do_GET()
            
            def _check_auth(self, auth_header: str) -> bool:
                """Check basic authentication"""
                try:
                    import base64
                    auth_type, auth_string = auth_header.split(' ', 1)
                    if auth_type.lower() != 'basic':
                        return False
                    
                    decoded = base64.b64decode(auth_string).decode('utf-8')
                    username, password = decoded.split(':', 1)
                    
                    return password == config.access_password
                    
                except Exception:
                    return False
            
            def end_headers(self):
                self.send_header('Server', 'Silent Steno Device')
                super().end_headers()
        
        return CustomHTTPHandler
    
    def add_file(self, file_path: str, share_filename: Optional[str] = None) -> bool:
        """Add file to HTTP share"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                logger.error(f"Source file does not exist: {file_path}")
                return False
            
            dest_filename = share_filename or source_path.name
            dest_path = Path(self.config.share_path) / dest_filename
            
            # Copy file to share directory
            shutil.copy2(source_path, dest_path)
            
            logger.info(f"Added file to HTTP share: {dest_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add file to HTTP share: {str(e)}")
            return False
    
    def remove_file(self, filename: str) -> bool:
        """Remove file from HTTP share"""
        try:
            file_path = Path(self.config.share_path) / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed file from HTTP share: {filename}")
                return True
            else:
                logger.warning(f"File not found in HTTP share: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove file from HTTP share: {str(e)}")
            return False
    
    def get_share_url(self) -> str:
        """Get HTTP share URL"""
        try:
            # Get local IP address
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            return f"http://{local_ip}:{self.config.http_port}/"
        except Exception:
            return f"http://localhost:{self.config.http_port}/"

class NetworkSharing:
    """Network file sharing coordinator"""
    
    def __init__(self, config: SharingConfig):
        self.config = config
        self.smb_server = SMBServer(config) if config.smb_enabled else None
        self.http_server = HTTPServer(config) if config.http_enabled else None
        self.access_log: List[Dict[str, Any]] = []
        
    def start_sharing(self) -> bool:
        """Start all enabled sharing services"""
        success = True
        
        if self.smb_server:
            if not self.smb_server.start():
                success = False
                logger.error("Failed to start SMB server")
        
        if self.http_server:
            if not self.http_server.start():
                success = False
                logger.error("Failed to start HTTP server")
        
        if success:
            logger.info("Network sharing started successfully")
        
        return success
    
    def stop_sharing(self) -> bool:
        """Stop all sharing services"""
        success = True
        
        if self.smb_server:
            if not self.smb_server.stop():
                success = False
        
        if self.http_server:
            if not self.http_server.stop():
                success = False
        
        if success:
            logger.info("Network sharing stopped")
        
        return success
    
    def share_file(self, file_path: str, share_filename: Optional[str] = None) -> Dict[str, str]:
        """Share file via all enabled methods"""
        results = {}
        
        if self.smb_server and self.smb_server.running:
            success = self.smb_server.add_file(file_path, share_filename)
            results['smb'] = 'success' if success else 'failed'
        
        if self.http_server and self.http_server.running:
            success = self.http_server.add_file(file_path, share_filename)
            results['http'] = self.http_server.get_share_url() if success else 'failed'
        
        # Log access
        if self.config.log_access:
            self.access_log.append({
                'timestamp': time.time(),
                'action': 'file_shared',
                'filename': share_filename or Path(file_path).name,
                'results': results
            })
        
        return results
    
    def unshare_file(self, filename: str) -> Dict[str, str]:
        """Remove file from all shares"""
        results = {}
        
        if self.smb_server and self.smb_server.running:
            success = self.smb_server.remove_file(filename)
            results['smb'] = 'success' if success else 'failed'
        
        if self.http_server and self.http_server.running:
            success = self.http_server.remove_file(filename)
            results['http'] = 'success' if success else 'failed'
        
        return results
    
    def get_sharing_status(self) -> Dict[str, Any]:
        """Get status of all sharing services"""
        status = {
            'smb_enabled': self.config.smb_enabled,
            'http_enabled': self.config.http_enabled,
            'smb_running': self.smb_server.running if self.smb_server else False,
            'http_running': self.http_server.running if self.http_server else False,
            'share_path': self.config.share_path,
            'files_shared': len(list(Path(self.config.share_path).glob('*'))) if Path(self.config.share_path).exists() else 0
        }
        
        if self.http_server and self.http_server.running:
            status['http_url'] = self.http_server.get_share_url()
        
        return status
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get access log"""
        return self.access_log.copy()
    
    def clear_share_directory(self) -> bool:
        """Clear all files from share directory"""
        try:
            share_path = Path(self.config.share_path)
            if share_path.exists():
                for file_path in share_path.glob('*'):
                    if file_path.is_file():
                        file_path.unlink()
                logger.info("Share directory cleared")
                return True
        except Exception as e:
            logger.error(f"Failed to clear share directory: {str(e)}")
            return False

def create_network_sharing(config: Optional[SharingConfig] = None) -> NetworkSharing:
    """Create network sharing with default or provided configuration"""
    if config is None:
        config = SharingConfig()
    
    return NetworkSharing(config)

def start_smb_server(config: Optional[SharingConfig] = None) -> SMBServer:
    """Convenience function to start SMB server"""
    if config is None:
        config = SharingConfig(http_enabled=False)
    
    server = SMBServer(config)
    server.start()
    return server

def start_http_server(config: Optional[SharingConfig] = None) -> HTTPServer:
    """Convenience function to start HTTP server"""
    if config is None:
        config = SharingConfig(smb_enabled=False)
    
    server = HTTPServer(config)
    server.start()
    return server

def configure_sharing(smb_enabled: bool = True, http_enabled: bool = True, 
                     share_path: str = "/tmp/silentSteno_share") -> SharingConfig:
    """Configure network sharing settings"""
    return SharingConfig(
        smb_enabled=smb_enabled,
        http_enabled=http_enabled,
        share_path=share_path
    )

def monitor_sharing_access(sharing: NetworkSharing) -> List[Dict[str, Any]]:
    """Monitor sharing access and return log entries"""
    return sharing.get_access_log()