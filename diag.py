import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_workflow_diagram():
    fig, ax = plt.figure(figsize=(12, 5)), plt.gca()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Styles
    box_props = dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#1565C0", lw=1.5)
    q_props = dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#2E7D32", lw=1.5)
    arrow_props = dict(arrowstyle="->", lw=1.5, color="#37474F")
    
    # Coordinates
    x_input, x_pre, x_enc, x_q, x_dec = 1, 3.5, 6, 8.5, 11
    y_main = 2.5

    # 1. Inputs
    ax.text(x_input, y_main+0.8, "EEG\n(2548)", ha="center", va="center", bbox=box_props, fontsize=10)
    ax.text(x_input, y_main-0.8, "ECG\n(10)", ha="center", va="center", bbox=box_props, fontsize=10)
    
    # 2. Preprocessing
    ax.text(x_pre, y_main, "Preprocessing\n(Norm + Concat)\nDim: 2558", ha="center", va="center", bbox=box_props, fontsize=10)
    
    # 3. Classical Encoder
    ax.text(x_enc, y_main, "Classical Encoder\n(Dense Layers)\nDim: 64", ha="center", va="center", bbox=box_props, fontsize=10)
    
    # 4. Quantum Layer
    ax.text(x_q, y_main, "Quantum Circuit\n(6 Qubits)\nRotations + QFT", ha="center", va="center", bbox=q_props, fontsize=10)

    # 5. Output/Tasks
    ax.text(x_dec, y_main+0.8, "Decoder\n(Reconstruction)", ha="center", va="center", bbox=box_props, fontsize=10)
    ax.text(x_dec, y_main-0.8, "Classifier\n(Emotion Label)", ha="center", va="center", bbox=box_props, fontsize=10)

    # Arrows
    # Inputs to Pre
    ax.annotate("", xy=(x_pre-0.8, y_main+0.2), xytext=(x_input+0.5, y_main+0.8), arrowprops=arrow_props)
    ax.annotate("", xy=(x_pre-0.8, y_main-0.2), xytext=(x_input+0.5, y_main-0.8), arrowprops=arrow_props)
    
    # Pre to Enc
    ax.annotate("", xy=(x_enc-0.8, y_main), xytext=(x_pre+0.8, y_main), arrowprops=arrow_props)
    
    # Enc to Quant
    ax.annotate("Latent z", xy=(x_q-0.8, y_main), xytext=(x_enc+0.8, y_main), 
                arrowprops=arrow_props, ha="center", va="bottom", fontsize=8)

    # Quant to Outputs
    ax.annotate("", xy=(x_dec-0.8, y_main+0.8), xytext=(x_q+0.8, y_main+0.2), arrowprops=arrow_props)
    ax.annotate("", xy=(x_dec-0.8, y_main-0.8), xytext=(x_q+0.8, y_main-0.2), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig("figures/workflow_diagram.pdf", bbox_inches='tight')
    plt.show()

create_workflow_diagram()