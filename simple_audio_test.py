#!/usr/bin/env python3
"""
Simple Audio Source Switcher - bypasses all the demo complexity
"""

import subprocess
import sys

class SimpleAudioSwitcher:
    def __init__(self):
        self.current_loopback = None
        
        # Your known devices
        self.sources = {
            'phone': {
                'name': 'Pixel 9 Pro',
                'mac': 'C0:1C:6A:AD:78:E6',
                'pa_source': 'bluez_source.C0_1C_6A_AD_78_E6.a2dp_source'
            }
        }
        
        self.headphones = {
            'name': 'Galaxy Buds3 Pro',
            'mac': 'BC:A0:80:EB:21:AA',
            'pa_sink': 'bluez_sink.BC_A0_80_EB_21_AA.a2dp_sink'
        }
    
    def stop_current_loopback(self):
        """Stop any current audio loopback"""
        try:
            result = subprocess.run("pactl list modules short | grep module-loopback", 
                                   shell=True, capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    module_id = line.split()[0]
                    print(f"Stopping loopback module {module_id}")
                    subprocess.run(f"pactl unload-module {module_id}", shell=True)
                    
        except Exception as e:
            print(f"Error stopping loopback: {e}")
    
    def start_audio_forwarding(self, source_key):
        """Start audio forwarding from source to headphones"""
        if source_key not in self.sources:
            print(f"Unknown source: {source_key}")
            return False
            
        source_info = self.sources[source_key]
        
        try:
            # Stop any existing loopback
            self.stop_current_loopback()
            
            # Create new loopback
            cmd = f"pactl load-module module-loopback source={source_info['pa_source']} sink={self.headphones['pa_sink']} latency_msec=40"
            
            print(f"Starting audio forwarding: {source_info['name']} -> {self.headphones['name']}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.current_loopback = result.stdout.strip()
                print(f"✅ Audio forwarding active")
                return True
            else:
                print(f"❌ Failed to start forwarding: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def check_devices(self):
        """Check if devices are connected"""
        print("\nChecking device status...")
        
        # Check sources
        result = subprocess.run("pactl list sources short", 
                               shell=True, capture_output=True, text=True)
        sources = result.stdout
        
        print("\nAvailable audio sources:")
        for key, info in self.sources.items():
            if info['pa_source'] in sources:
                print(f"  ✅ {info['name']} - Connected")
            else:
                print(f"  ❌ {info['name']} - Not available")
        
        # Check sink
        result = subprocess.run("pactl list sinks short", 
                               shell=True, capture_output=True, text=True)
        sinks = result.stdout
        
        print(f"\nAudio output:")
        if self.headphones['pa_sink'] in sinks:
            print(f"  ✅ {self.headphones['name']} - Connected")
        else:
            print(f"  ❌ {self.headphones['name']} - Not available")
    
    def show_status(self):
        """Show current audio forwarding status"""
        try:
            result = subprocess.run("pactl list modules short | grep module-loopback", 
                                   shell=True, capture_output=True, text=True)
            
            if result.stdout.strip():
                print("\n🔊 Audio forwarding is ACTIVE")
                
                # Show source outputs
                result = subprocess.run("pactl list source-outputs short", 
                                       shell=True, capture_output=True, text=True)
                print("Source outputs:", result.stdout.strip())
                
                # Show sink inputs  
                result = subprocess.run("pactl list sink-inputs short", 
                                       shell=True, capture_output=True, text=True)
                print("Sink inputs:", result.stdout.strip())
            else:
                print("\n⚫ No audio forwarding active")
                
        except Exception as e:
            print(f"Error checking status: {e}")
    
    def run_menu(self):
        """Simple menu interface"""
        while True:
            print("\n" + "="*50)
            print("Simple Audio Source Switcher")
            print("="*50)
            print("1. Check device status")
            print("2. Start phone -> buds forwarding")
            print("3. Stop audio forwarding") 
            print("4. Show current status")
            print("5. Play test sound to buds")
            print("0. Exit")
            print("-"*50)
            
            choice = input("Choose option (0-5): ").strip()
            
            if choice == '1':
                self.check_devices()
                
            elif choice == '2':
                self.start_audio_forwarding('phone')
                
            elif choice == '3':
                self.stop_current_loopback()
                print("✅ Audio forwarding stopped")
                
            elif choice == '4':
                self.show_status()
                
            elif choice == '5':
                print("Playing test sound to Galaxy Buds...")
                subprocess.run(f"paplay -d {self.headphones['pa_sink']} /usr/share/sounds/alsa/Front_Left.wav", shell=True)
                
            elif choice == '0':
                self.stop_current_loopback()
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice")

def main():
    switcher = SimpleAudioSwitcher()
    try:
        switcher.run_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
        switcher.stop_current_loopback()

if __name__ == "__main__":
    main()
