#!/bin/bash

# Kill any running Silent Steno demo instances

echo "Killing any running Silent Steno demo instances..."

# Kill by process name
pkill -f "minimal_demo_refactored.py" 2>/dev/null && echo "✅ Killed minimal_demo_refactored.py"
pkill -f "minimal_demo.py" 2>/dev/null && echo "✅ Killed minimal_demo.py"
pkill -f "demo_live_session.py" 2>/dev/null && echo "✅ Killed demo_live_session.py"

# Kill any Python processes that might be Kivy apps
ps aux | grep -i kivy | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null && echo "✅ Killed Kivy processes"

echo "✅ Cleanup complete!"
echo "You can now start the demo fresh."