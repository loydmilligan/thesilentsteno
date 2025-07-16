# Silent Steno Testing Guide

## Current System State

**Project Status:** The Silent Steno is in **Phase 2** of development with 15 completed tasks. The system has a complete UI framework, data management, export capabilities, and application integration layer.

### ✅ Real Features (Fully Implemented)

**1. Database System**
- SQLite database with full session management
- SQLAlchemy ORM models for sessions, transcripts, participants
- Database migrations and backup system
- Real data persistence and retrieval

**2. Export System**
- PDF generation with ReportLab (creates actual PDFs)
- Email export with SMTP configuration
- USB drive detection and file transfer
- Network sharing (SMB/HTTP servers)
- Multiple export formats (PDF, TXT, JSON, HTML)

**3. Touch UI Framework**
- Complete Kivy-based touch interface
- Responsive layouts optimized for 3.5-5" screens
- Touch controls with visual feedback
- Theme system with light/dark modes
- Session management workflows

**4. Application Integration**
- Central application controller
- Event-driven communication system
- Configuration management with hot-reload
- Comprehensive logging system
- Error handling and recovery mechanisms
- Performance monitoring

**5. Configuration Management**
- JSON-based configuration files
- Environment-specific settings
- Hot-reload capabilities
- Validation and error checking

### ❌ Mock/Simulated Features

**1. Audio Capture**
- No real Bluetooth audio pipeline yet
- Audio visualizer uses simulated data
- No actual audio recording or processing

**2. AI Processing**
- No real Whisper transcription
- No actual LLM analysis
- Transcript display uses sample text
- Speaker detection is simulated

**3. Hardware Integration**
- No actual Pi 5 audio routing
- No real Bluetooth device management
- System monitoring uses mock data

**4. Session Recording**
- Sessions are simulated with sample data
- No actual meeting audio capture
- Demo modes provide realistic workflows

---

## Hardware Requirements

### Minimum Requirements
- **Platform:** Raspberry Pi 5 (recommended) or Linux desktop
- **Display:** 3.5-5" touchscreen or standard monitor
- **Memory:** 4GB RAM minimum (8GB recommended)
- **Storage:** 16GB+ SD card or SSD
- **Python:** 3.8+ with pip

### Recommended Pi 5 Setup
- **OS:** Raspberry Pi OS (64-bit)
- **Display:** Official 7" touchscreen or compatible
- **Case:** Touchscreen-compatible case
- **Power:** Official Pi 5 power supply

### Dependencies
```bash
# System packages
sudo apt update
sudo apt install python3-dev python3-pip python3-kivy sqlite3

# Python packages (auto-installed by app)
pip3 install kivy sqlalchemy reportlab psutil pydantic watchdog
```

---

## Testing Instructions

### 1. Environment Setup

```bash
# Navigate to project directory
cd /home/mmariani/projects/thesilentsteno

# Verify Python environment
python3 --version  # Should be 3.8+

# Check dependencies
python3 -c "import kivy; print('Kivy OK')"
python3 -c "import sqlalchemy; print('SQLAlchemy OK')"
```

### 2. Integration Tests

**Run the full integration test suite:**
```bash
python3 test_integration.py
```

**Expected Output:**
```
=== Task-6.1 Application Integration Layer Tests ===

Testing core module imports...
✓ All core imports successful

Testing application creation...
✓ Application created successfully
✓ Application initialized successfully

Testing event system...
✓ Event bus created
✓ Event subscription created
✓ Event received successfully

Testing configuration system...
✓ Configuration loaded
✓ Configuration access working

Testing component registry...
✓ Component registry created
✓ Component registered
✓ Component retrieved successfully

Testing logging system...
✓ Logging system setup
✓ Logger created
✓ Log message sent

Testing error handling...
✓ Error handler created
✓ Error handled successfully
✓ Error record created correctly

=== Test Results: 7/7 passed ===
🎉 All integration tests passed!
```

### 3. Demo Applications

#### A. Simple UI Demo
```bash
python3 demo_simple.py
```

**What You'll See:**
- Basic session interface
- Start/Stop controls
- Status indicators
- Simple workflow demonstration

**Features Tested:**
- Touch UI responsiveness
- Session state management
- Basic navigation

#### B. Live Session Demo (Recommended)
```bash
python3 demo_live_session.py
```

**What You'll See:**
- Full meeting interface with multiple screens
- Live transcript display with speaker labels
- Real-time audio visualizer
- Session controls and status indicators
- Menu system with component demos

**Navigation:**
- Touch buttons to switch between demos
- ESC key or back button to return to menu
- Full touch gesture support

**Demo Screens:**
1. **Full Live Session** - Complete meeting interface
2. **Session Controls** - Start/Stop/Pause buttons
3. **Transcript Display** - Scrolling text with speakers
4. **Audio Visualizer** - Real-time audio bars
5. **Status Indicators** - System health display

#### C. Touch UI Demo
```bash
python3 demo_touch_ui.py
```

**What You'll See:**
- Touch-optimized controls
- Gesture recognition
- Visual feedback
- Haptic responses (if supported)

### 4. Database Testing

**Test database operations:**
```bash
# Python interactive session
python3 -c "
from src.data.database import DatabaseManager
from src.data.models import Session, TranscriptEntry

# Create database connection
db = DatabaseManager()
db.init_database()

# Create test session
session = Session(title='Test Meeting', date='2024-01-01')
db.add_session(session)

print('Database test successful!')
"
```

### 5. Export System Testing

**Test PDF generation:**
```bash
python3 -c "
from src.export.pdf_generator import PDFGenerator
from src.data.models import Session

# Create sample session
session = Session(title='Test Export', date='2024-01-01')

# Generate PDF
pdf_gen = PDFGenerator()
pdf_path = pdf_gen.generate_session_pdf(session)

print(f'PDF generated: {pdf_path}')
"
```

**Test email export:**
```bash
python3 -c "
from src.export.email_exporter import EmailExporter
from src.data.models import Session

# Create sample session
session = Session(title='Test Email', date='2024-01-01')

# Setup email exporter (requires SMTP config)
email_exporter = EmailExporter()
# email_exporter.send_session_email(session, 'test@example.com')

print('Email export system loaded successfully')
"
```

---

## Expected Behavior

### Demo Applications
1. **Startup Time:** 3-5 seconds on Pi 5
2. **UI Responsiveness:** Smooth touch interactions
3. **Visual Quality:** Clear text and icons on touchscreen
4. **Navigation:** Intuitive menu system

### Performance Metrics
- **Memory Usage:** 150-300MB during demos
- **CPU Usage:** 15-40% during active visualization
- **Storage:** ~50MB for application files
- **Database:** Fast queries (<100ms)

### Visual Indicators
- **Green Status:** Systems operational
- **Blue Progress:** Loading/processing states
- **Red Alerts:** Error conditions
- **Animations:** Smooth transitions and updates

---

## Hardware-Specific Testing

### Pi 5 Touchscreen Setup
```bash
# Configure display resolution
sudo nano /boot/config.txt
# Add/modify:
# hdmi_force_hotplug=1
# hdmi_group=2
# hdmi_mode=87
# hdmi_cvt=800 480 60 6 0 0 0

# Configure touch calibration
sudo apt install xinput-calibrator
xinput_calibrator
```

### Full Screen Mode
```bash
# Edit demo files to enable fullscreen
# In demo_live_session.py, change:
# Config.set('graphics', 'fullscreen', '1')
```

### Performance Optimization
```bash
# Increase GPU memory
sudo raspi-config
# Advanced Options > Memory Split > 128

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-power-save
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
sudo apt install python3-kivy python3-dev
pip3 install -r requirements.txt
```

**2. Display Issues**
```bash
# Check display configuration
tvservice -s
# Adjust resolution in /boot/config.txt
```

**3. Touch Not Working**
```bash
# Check touch device
ls /dev/input/event*
# Calibrate touch
xinput_calibrator
```

**4. Performance Issues**
```bash
# Monitor system resources
htop
# Check temperature
vcgencmd measure_temp
```

### Debug Mode
```bash
# Run with debug logging
KIVY_LOG_LEVEL=debug python3 demo_live_session.py
```

### Log Files
- **Application logs:** `logs/silentst.log`
- **System logs:** `/var/log/syslog`
- **Kivy logs:** `~/.kivy/logs/`

---

## Development Testing

### Code Quality
```bash
# Check Python syntax
python3 -m py_compile src/core/*.py
python3 -m py_compile src/ui/*.py

# Run basic imports
python3 -c "import src.core; print('Core module OK')"
python3 -c "import src.ui; print('UI module OK')"
```

### Performance Testing
```bash
# Memory profiling
python3 -c "
import tracemalloc
tracemalloc.start()
# Run your code here
current, peak = tracemalloc.get_traced_memory()
print(f'Memory usage: {current/1024/1024:.1f}MB')
"
```

---

## Next Steps

### Immediate Testing Goals
1. **UI Validation** - Test all demo applications
2. **Touch Responsiveness** - Verify gesture support
3. **Data Persistence** - Test database operations
4. **Export Functions** - Generate actual PDFs

### Future Development
1. **Audio Pipeline** - Bluetooth audio capture
2. **AI Integration** - Whisper transcription
3. **Hardware Integration** - Pi 5 optimization
4. **Production Deployment** - Automated setup

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files for error messages
3. Test on desktop Linux before Pi 5
4. Verify all dependencies are installed

**Remember:** This is a development system with mock audio/AI features. The UI and data management are fully functional for testing the user experience and hardware integration.