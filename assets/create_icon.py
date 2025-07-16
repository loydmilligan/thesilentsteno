#!/usr/bin/env python3
"""
Create a simple icon for Silent Steno using PIL.
Run this to generate the desktop icon.
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import os
except ImportError:
    print("PIL not installed. Install with: pip install pillow")
    exit(1)

def create_icon():
    # Create a 128x128 icon
    size = 128
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Background circle - dark blue
    margin = 8
    draw.ellipse([margin, margin, size-margin, size-margin], 
                fill=(45, 85, 135, 255), outline=(70, 130, 180, 255), width=3)
    
    # Microphone body - silver/gray
    mic_width = 24
    mic_height = 36
    mic_x = size//2 - mic_width//2
    mic_y = size//2 - mic_height//2 - 8
    
    # Microphone capsule (top rounded part)
    draw.rounded_rectangle([mic_x, mic_y, mic_x + mic_width, mic_y + mic_height], 
                          radius=12, fill=(200, 200, 200, 255), outline=(160, 160, 160, 255), width=2)
    
    # Microphone grille lines
    for i in range(3):
        y = mic_y + 8 + i * 6
        draw.line([mic_x + 6, y, mic_x + mic_width - 6, y], fill=(120, 120, 120, 255), width=2)
    
    # Microphone stand
    stand_x = size//2 - 2
    stand_y = mic_y + mic_height
    draw.rectangle([stand_x, stand_y, stand_x + 4, stand_y + 12], fill=(160, 160, 160, 255))
    
    # Base
    base_width = 20
    base_x = size//2 - base_width//2
    base_y = stand_y + 12
    draw.rectangle([base_x, base_y, base_x + base_width, base_y + 4], fill=(120, 120, 120, 255))
    
    # Sound waves
    wave_center_x = size//2 + mic_width//2 + 8
    wave_center_y = size//2 - 4
    
    for i in range(3):
        radius = 15 + i * 8
        alpha = max(100 - i * 30, 40)
        # Draw arc for sound waves
        bbox = [wave_center_x - radius, wave_center_y - radius, 
                wave_center_x + radius, wave_center_y + radius]
        draw.arc(bbox, start=-30, end=30, fill=(255, 255, 255, alpha), width=3)
    
    # AI indicator (small dot)
    ai_x = size - 25
    ai_y = 25
    draw.ellipse([ai_x - 6, ai_y - 6, ai_x + 6, ai_y + 6], 
                fill=(0, 255, 100, 255), outline=(0, 200, 80, 255), width=2)
    
    return img

if __name__ == '__main__':
    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Create and save icon
    icon = create_icon()
    icon.save('assets/icon.png', 'PNG')
    print("✅ Icon created at assets/icon.png")
    
    # Also create smaller versions
    icon.resize((64, 64), Image.Resampling.LANCZOS).save('assets/icon-64.png', 'PNG')
    icon.resize((32, 32), Image.Resampling.LANCZOS).save('assets/icon-32.png', 'PNG')
    print("✅ Additional icon sizes created")