# Other Directories Documentation

## Overview

This document covers the additional directories in The Silent Steno project that contain utility scripts, assets, and database migration files. These components support the main application with icon generation, database schema management, and other auxiliary functionality.

## Dependencies

### External Dependencies
- `PIL` (Pillow) - Python Imaging Library for icon generation
- `SQLAlchemy` - Database ORM
- `alembic` - Database migration tool
- `logging` - Python logging
- `os` - Operating system interface

### Internal Dependencies
- `src.data.models` - Database models and schema definitions

## Directory Structure

```
├── assets/
│   ├── create_icon.py          # Icon generation script
│   ├── icon.png                # Generated application icon (128x128)
│   ├── icon-64.png             # Medium icon (64x64)
│   └── icon-32.png             # Small icon (32x32)
├── migrations/
│   ├── env.py                  # Alembic migration environment
│   └── versions/               # Database migration versions
```

## File Documentation

### Assets Directory

#### 1. `create_icon.py`

**Purpose**: Programmatic generation of application icons for The Silent Steno with custom graphics representing a microphone and AI processing.

##### Functions

###### `create_icon()`
Creates a 128x128 pixel application icon with custom graphics.

**Returns:**
- `PIL.Image` - Generated icon image

**Design Elements:**
- **Background**: Dark blue circular background with gradient border
- **Microphone**: Silver/gray microphone with detailed grille pattern
- **Stand**: Microphone stand with base
- **Sound Waves**: Animated-style sound waves emanating from microphone
- **AI Indicator**: Green dot indicating AI processing capability

**Visual Components:**
```python
# Background circle - dark blue
draw.ellipse([margin, margin, size-margin, size-margin], 
            fill=(45, 85, 135, 255), outline=(70, 130, 180, 255), width=3)

# Microphone body - silver/gray with rounded corners
draw.rounded_rectangle([mic_x, mic_y, mic_x + mic_width, mic_y + mic_height], 
                      radius=12, fill=(200, 200, 200, 255), outline=(160, 160, 160, 255), width=2)

# Microphone grille lines for realistic appearance
for i in range(3):
    y = mic_y + 8 + i * 6
    draw.line([mic_x + 6, y, mic_x + mic_width - 6, y], fill=(120, 120, 120, 255), width=2)

# Sound waves with transparency gradient
for i in range(3):
    radius = 15 + i * 8
    alpha = max(100 - i * 30, 40)
    draw.arc(bbox, start=-30, end=30, fill=(255, 255, 255, alpha), width=3)

# AI indicator - bright green dot
draw.ellipse([ai_x - 6, ai_y - 6, ai_x + 6, ai_y + 6], 
            fill=(0, 255, 100, 255), outline=(0, 200, 80, 255), width=2)
```

**Usage Example:**
```python
# Generate application icons
from assets.create_icon import create_icon
import os

# Create assets directory
os.makedirs('assets', exist_ok=True)

# Generate icon
icon = create_icon()

# Save in multiple sizes
icon.save('assets/icon.png', 'PNG')  # 128x128
icon.resize((64, 64), Image.Resampling.LANCZOS).save('assets/icon-64.png', 'PNG')
icon.resize((32, 32), Image.Resampling.LANCZOS).save('assets/icon-32.png', 'PNG')

print("✅ Icons generated successfully")
```

**Generated Files:**
- `icon.png` - Main application icon (128x128)
- `icon-64.png` - Medium size icon for various UI elements
- `icon-32.png` - Small size icon for system tray, notifications

**Design Rationale:**
- **Microphone Symbol**: Clearly represents the audio recording functionality
- **AI Indicator**: Green dot signifies AI processing capabilities
- **Sound Waves**: Visual representation of audio processing
- **Professional Aesthetic**: Clean, modern design suitable for desktop applications
- **Scalability**: Vector-style approach ensures clarity at different sizes

### Migrations Directory

#### 1. `env.py`

**Purpose**: Alembic migration environment configuration for database schema management and versioning.

##### Functions

###### `run_migrations_offline()`
Runs database migrations in offline mode (without active database connection).

**Features:**
- Generates SQL migration scripts
- Useful for deployment scenarios
- No active database connection required

**Implementation:**
```python
def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()
```

###### `run_migrations_online()`
Runs database migrations in online mode (with active database connection).

**Features:**
- Executes migrations directly against database
- Real-time migration execution
- Connection pooling with NullPool for safety

**Implementation:**
```python
def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()
```

**Configuration Elements:**
- **Target Metadata**: References `src.data.models.Base.metadata` for schema definitions
- **Logging Configuration**: Integrates with Python logging system
- **Path Resolution**: Dynamically resolves paths to include source modules

**Database Migration Workflow:**
```bash
# Initialize migration environment (done once)
alembic init migrations

# Generate new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1

# Check migration status
alembic current
alembic history
```

**Integration with Data Models:**
```python
# env.py imports the database models
from src.data.models import Base

# Target metadata points to SQLAlchemy Base
target_metadata = Base.metadata
```

**Migration File Structure:**
```
migrations/
├── env.py                      # Environment configuration
├── alembic.ini                 # Alembic configuration file
├── script.py.mako              # Migration script template
└── versions/                   # Migration version files
    ├── 001_initial_schema.py   # Initial database schema
    ├── 002_add_sessions.py     # Session table addition
    └── 003_add_transcripts.py  # Transcript table addition
```

## Usage Patterns

### Icon Generation Workflow
```python
# Complete icon generation script
#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from assets.create_icon import create_icon
    from PIL import Image
    
    def generate_all_icons():
        """Generate all required icon sizes"""
        print("🎨 Generating Silent Steno icons...")
        
        # Create assets directory
        assets_dir = project_root / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # Generate base icon
        icon = create_icon()
        
        # Save multiple sizes
        sizes = [
            (128, "icon.png"),
            (64, "icon-64.png"),
            (32, "icon-32.png"),
            (16, "icon-16.png")  # Additional tiny icon
        ]
        
        for size, filename in sizes:
            if size == 128:
                # Save original size
                icon.save(assets_dir / filename, 'PNG')
            else:
                # Resize and save
                resized = icon.resize((size, size), Image.Resampling.LANCZOS)
                resized.save(assets_dir / filename, 'PNG')
            
            print(f"✅ Created {filename} ({size}x{size})")
        
        print("🎉 Icon generation complete!")
        
    if __name__ == "__main__":
        generate_all_icons()
        
except ImportError as e:
    print(f"❌ Error: {e}")
    print("📦 Install required dependencies: pip install pillow")
    sys.exit(1)
```

### Database Migration Management
```python
# Database migration utilities
import subprocess
import sys
from pathlib import Path

class MigrationManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.migrations_dir = project_root / "migrations"
        
    def init_migrations(self):
        """Initialize Alembic migrations"""
        try:
            subprocess.run([
                sys.executable, "-m", "alembic", "init", "migrations"
            ], check=True, cwd=self.project_root)
            print("✅ Migration environment initialized")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to initialize migrations: {e}")
            
    def generate_migration(self, message: str):
        """Generate new migration"""
        try:
            subprocess.run([
                sys.executable, "-m", "alembic", "revision", 
                "--autogenerate", "-m", message
            ], check=True, cwd=self.project_root)
            print(f"✅ Migration generated: {message}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate migration: {e}")
            
    def upgrade_database(self):
        """Apply all pending migrations"""
        try:
            subprocess.run([
                sys.executable, "-m", "alembic", "upgrade", "head"
            ], check=True, cwd=self.project_root)
            print("✅ Database upgraded successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upgrade database: {e}")
            
    def downgrade_database(self, revision: str = "-1"):
        """Rollback database migrations"""
        try:
            subprocess.run([
                sys.executable, "-m", "alembic", "downgrade", revision
            ], check=True, cwd=self.project_root)
            print(f"✅ Database downgraded to {revision}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to downgrade database: {e}")
            
    def migration_status(self):
        """Check migration status"""
        try:
            # Current revision
            current = subprocess.run([
                sys.executable, "-m", "alembic", "current"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Migration history
            history = subprocess.run([
                sys.executable, "-m", "alembic", "history"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            print("📊 Migration Status:")
            print(f"Current: {current.stdout.strip()}")
            print(f"History:\n{history.stdout}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to get migration status: {e}")

# Usage example
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    migration_manager = MigrationManager(project_root)
    
    # Check status
    migration_manager.migration_status()
    
    # Generate migration if needed
    # migration_manager.generate_migration("Add user preferences table")
    
    # Apply migrations
    # migration_manager.upgrade_database()
```

## Integration with Main Application

### Icon Usage in Desktop Integration
```python
# Desktop file generation for Linux
def create_desktop_file():
    """Create desktop entry for The Silent Steno"""
    desktop_content = f"""[Desktop Entry]
Name=The Silent Steno
Comment=Bluetooth AI Meeting Recorder
Exec={sys.executable} /path/to/main.py
Icon=/path/to/assets/icon.png
Terminal=false
Type=Application
Categories=AudioVideo;Audio;Recorder;
StartupNotify=true
"""
    
    desktop_path = Path.home() / ".local/share/applications/silent-steno.desktop"
    desktop_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(desktop_path, 'w') as f:
        f.write(desktop_content)
    
    # Make executable
    desktop_path.chmod(0o755)
    print(f"✅ Desktop file created: {desktop_path}")
```

### Database Schema Evolution
```python
# Example migration version file structure
"""Add session metadata

Revision ID: 12345abcde
Revises: previous_revision
Create Date: 2024-01-15 10:30:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '12345abcde'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None

def upgrade():
    # Add new columns or tables
    op.add_column('sessions', sa.Column('metadata', sa.JSON, nullable=True))
    op.add_column('sessions', sa.Column('quality_score', sa.Float, nullable=True))

def downgrade():
    # Reverse the changes
    op.drop_column('sessions', 'metadata')
    op.drop_column('sessions', 'quality_score')
```

## Summary

The assets and migrations directories provide essential support functionality:

1. **Assets Directory**: Programmatic icon generation with professional design
2. **Migrations Directory**: Database schema management and versioning
3. **Integration**: Seamless integration with main application systems
4. **Automation**: Scripted workflows for icon generation and database management

These utilities ensure The Silent Steno has a professional appearance and robust database management capabilities essential for a production application.