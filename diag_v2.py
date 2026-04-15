import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os

def create_premium_diagram():
    # Paths to the images
    signals_path = "figures/combined_signals.png"
    circuit_path = "figures/quantum_circuit.png"

    # Set up the figure
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.5, 10)
    ax.axis('off')

    # Color Palette
    COLOR_RED = "#D32F2F"
    COLOR_GREY = "#455A64"
    COLOR_BLUE = "#1976D2"
    
    def add_image(path, x, y, zoom=0.1):
        if os.path.exists(path):
            try:
                from PIL import Image
                import numpy as np
                img_pil = Image.open(path)
                if img_pil.mode in ("RGBA", "P"):
                    img_pil = img_pil.convert("RGBA")
                img = np.array(img_pil)
                
                imagebox = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                ax.add_artist(ab)
            except Exception as e:
                print(f"Warning: Could not load image {path}: {e}")
                ax.text(x, y, "[Image Load Error]", ha='center', va='center', fontsize=8, color='red')

    # 1. Flow Headings (Above the components)
    heading_y = 7.5
    ax.text(2.75, heading_y, "INPUT", color=COLOR_RED, fontsize=24, fontweight='bold', ha='center')
    ax.text(7.5, heading_y, "PROCESS", color=COLOR_GREY, fontsize=24, fontweight='bold', ha='center')
    ax.text(12.25, heading_y, "OUTPUT", color=COLOR_BLUE, fontsize=24, fontweight='bold', ha='center')

    # 2. Core Components
    comp_y = 5.8 # Center vertical position
    
    # Balance zooms to make images look similar in scale
    add_image(signals_path, 2.4, comp_y, zoom=0.2) # Adjusted for uniformity
    add_image(circuit_path, 7.5, comp_y, zoom=0.08)  # Adjusted for uniformity
    
    # Output: Result visualization
    ax.add_patch(patches.RegularPolygon((12.25, comp_y), 6, radius=0.8, color=COLOR_BLUE, alpha=0.15))
    ax.text(12.25, comp_y, "EMOTION\nPREDICTION", ha='center', va='center', fontsize=16, fontweight='bold', color=COLOR_BLUE)

    # 3. Connections
    # Horizontal flow arrows
    ax.annotate("", xy=(5.0, comp_y), xytext=(4.3, comp_y), arrowprops=dict(arrowstyle="->", lw=2.5, color=COLOR_GREY, alpha=0.6))
    ax.annotate("", xy=(10.7, comp_y), xytext=(10.0, comp_y), arrowprops=dict(arrowstyle="->", lw=2.5, color=COLOR_GREY, alpha=0.6))

    # Vertical Callout Arrows (RESTORED)
    ax.annotate("", xytext=(2.75, comp_y - 1.4), xy=(2.75, 3.2), arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=2, alpha=0.8))
    ax.annotate("", xytext=(7.5, comp_y - 1.4), xy=(7.5, 3.2), arrowprops=dict(arrowstyle="->", color=COLOR_GREY, lw=2, alpha=0.8))
    ax.annotate("", xytext=(12.25, comp_y - 1.4), xy=(12.25, 3.2), arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=2, alpha=0.8))

    # 4. Detail Boxes (Bottom - ORIGINAL ORIENTATION)
    def draw_detail_box(x, y, title, details, color):
        width = 4.2
        height = 2.2
        # Restore original centered orientation but with restored dimensions
        box = patches.FancyBboxPatch((x - width/2, y - height/2), width, height, boxstyle="round,pad=0.2", 
                                     facecolor="white", edgecolor=color, lw=2.5, alpha=1)
        ax.add_patch(box)
        ax.text(x, y + 0.4, title, ha='center', va='center', fontsize=18, fontweight='bold', color=color)
        ax.text(x, y - 0.4, details, ha='center', va='center', fontsize=14, linespacing=1.6)

    box_y = 1.6
    draw_detail_box(2.75, box_y, "Multimodal Fusion", "EEG: 2548 Features\nECG: 10 Statistical Features\nDim: 2558 Normalized", COLOR_RED)
    draw_detail_box(7.5, box_y, "Hybrid QVAE", "6-Qubit PQC Processing\nRing Entanglement\nFourier-Domain Mapping", COLOR_GREY)
    draw_detail_box(12.25, box_y, "Performance", "MSE Loss: 0.2911\nAccuracy: 85.71%\nStability (σ: 0.0164)", COLOR_BLUE)

    # 5. Top Context Labels
    subtitle_y = 8.5
    ax.text(2.75, subtitle_y, "Raw Physiological Signals", ha='center', fontsize=14, alpha=0.7)
    ax.text(7.5, subtitle_y, "Latent Space Transformation", ha='center', fontsize=14, alpha=0.7)
    ax.text(12.25, subtitle_y, "Classification & Inference", ha='center', fontsize=14, alpha=0.7)

    plt.savefig("figures/arch_premium.png", bbox_inches='tight', transparent=False, facecolor='white')
    plt.savefig("figures/arch_premium.pdf", bbox_inches='tight')
    print("Saved optimized layout: figures/arch_premium.png and .pdf")

if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    create_premium_diagram()
