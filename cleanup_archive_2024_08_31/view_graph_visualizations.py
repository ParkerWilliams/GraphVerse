#!/usr/bin/env python3
"""
Simple script to display the generated graph visualizations.

Usage:
    python view_graph_visualizations.py [layout]
    
Where layout can be: spring, circular, spectral, summary, or all (default)
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
from pathlib import Path

def show_image(filename):
    """Display a single image file."""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return False
        
    try:
        img = mpimg.imread(filename)
        plt.figure(figsize=(16, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"GraphVerse Small Graph - {Path(filename).stem.replace('small_graph_', '').replace('_', ' ').title()}")
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        print(f"❌ Error displaying {filename}: {e}")
        return False

def main():
    """Main function to display graph visualizations."""
    layouts = ['spring', 'circular', 'spectral', 'summary']
    
    # Parse command line argument
    if len(sys.argv) > 1:
        requested = sys.argv[1].lower()
        if requested == 'all':
            show_layouts = layouts
        elif requested in layouts:
            show_layouts = [requested]
        else:
            print(f"❌ Invalid layout: {requested}")
            print(f"Available layouts: {', '.join(layouts)}, all")
            sys.exit(1)
    else:
        show_layouts = layouts  # Show all by default
    
    print("🖼️ GraphVerse Small Graph Visualizations")
    print("=" * 50)
    
    success_count = 0
    for layout in show_layouts:
        filename = f"small_graph_{layout}.png"
        print(f"📊 Displaying: {filename}")
        
        if show_image(filename):
            success_count += 1
            print(f"✅ Successfully displayed {filename}")
        else:
            print(f"❌ Failed to display {filename}")
    
    print(f"\n🎉 Displayed {success_count}/{len(show_layouts)} visualizations")
    
    if success_count > 0:
        print("\n📝 Graph Legend:")
        print("   🔴 Red nodes: Ascender rules (10 nodes)")
        print("   🟢 Teal nodes: Even rules (15 nodes)")  
        print("   ⚫ Gray nodes: Regular nodes (60 nodes)")
        print("   🟠 Orange-Purple nodes: Repeater rules (15 nodes)")
        print("      • Color varies by k-value (2→12)")
        print("      • Thick colored edges show k-cycles")
        print(f"\n📊 Graph has 100 nodes, 3,979 edges, 40.2% density")
        print(f"   37 total repeater cycles with average length 9.4")

if __name__ == "__main__":
    main()