import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from thing import (
    create_harmonic_potential, create_inf_square_well,
    sudden_evolution, dyson_evolution, adiabatic_theorem_evolution,
    H_linear, H_infwell,
    main as quantum_main
)

# Set page to full width
st.set_page_config(layout="wide")

def create_animation(psi_xt, x, times, c_m, V, rate=10, only_density_and_pop=False, model_names=None):
    """Create animation for Streamlit display. If only_density_and_pop is True, only plot probability density and populations."""
    if only_density_and_pop:
        # 'all' model: c_m is a list of arrays, do not convert
        n_models = len(psi_xt)
        # Shared |psi|^2 y-axis for all models
        prob_max_shared = max([np.max(np.abs(psi_xt[m])**2) for m in range(n_models)]) * 1.1
        V_max_list = [np.max(V[m] if V[m].ndim > 1 else V[m]) * 1.1 for m in range(n_models)]
        M = min(10, min([c_m[m].shape[1] for m in range(n_models)]))
        fig = make_subplots(
            rows=2, cols=n_models,
            subplot_titles=[
                f"{name} | Probability Density" for name in model_names
            ] + [
                f"{name} | State Populations" for name in model_names
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": True} for _ in range(n_models)],
                   [{"secondary_y": False} for _ in range(n_models)]]
        )
        frames = []
        for i in range(len(times[0])):
            frame_data = []
            for m in range(n_models):
                psi = psi_xt[m][i]
                V_frame = V[m][i] if V[m].ndim > 1 else V[m]
                Pm = np.abs(c_m[m][i][:M])**2
                # Order: |psi|^2 (row=1,col=m+1), V (row=1,col=m+1,secondary_y), Pm (row=2,col=m+1)
                frame_data.append(go.Scatter(x=x, y=np.abs(psi)**2, name=f"|ψ|² {model_names[m]}", showlegend=False))
                frame_data.append(go.Scatter(x=x, y=V_frame, name=f"Potential {model_names[m]}", line=dict(dash='dash'), showlegend=False))
                frame_data.append(go.Bar(x=np.arange(M), y=Pm, name=f"Populations {model_names[m]}", showlegend=False))
            frames.append(go.Frame(data=frame_data, name=f"frame{i}"))
        # Add initial traces in the same order as frame_data
        for m in range(n_models):
            fig.add_trace(
                go.Scatter(x=x, y=np.abs(psi_xt[m][0])**2, name=f"|ψ|² {model_names[m]}", showlegend=False),
                row=1, col=m+1, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=x, y=V[m][0] if V[m].ndim > 1 else V[m], name=f"Potential {model_names[m]}", line=dict(dash='dash'), showlegend=False),
                row=1, col=m+1, secondary_y=True
            )
            fig.add_trace(
                go.Bar(x=np.arange(M), y=np.abs(c_m[m][0][:M])**2, name=f"Populations {model_names[m]}", showlegend=False),
                row=2, col=m+1
            )
        fig.update_layout(
            height=700,
            showlegend=False,
            width=None,
            sliders=[{
                'currentvalue': {
                    'prefix': 'Time: ',
                    'suffix': ' s',
                    'xanchor': 'right',
                    'font': {'size': 28}
                },
                'pad': {'t': 50},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top',
                'transition': {'duration': 0},
                'steps': [
                    {
                        'label': f'{t:.2f}',
                        'method': 'animate',
                        'args': [[f'frame{i}'], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                    for i, t in enumerate(times[0])
                ]
            }],
            updatemenus=[
                {
                    'type': 'buttons',
                    'showactive': False,
                    'x': 0.1,
                    'y': -.15,
                    'xanchor': 'right',
                    'yanchor': 'top',
                    'pad': {'t': 0, 'r': 10},
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 100, 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                },
                {
                    'type': 'buttons',
                    'direction': 'right',
                    'showactive': True,
                    'x': 0.1,
                    'y': -.3,
                    'xanchor': 'left',
                    'yanchor': 'top',
                    'pad': {'t': 0, 'r': 10},
                    'buttons': [
                        {
                            'label': f'{speed}x',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 100/(speed), 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                        for speed in [1, 5, 10, 20, 50]
                    ]
                }
            ]
        )
        for m in range(n_models):
            fig.update_yaxes(title_text="|ψ|²", range=[0, prob_max_shared], row=1, col=m+1, secondary_y=False)
            fig.update_yaxes(title_text="V", range=[0, V_max_list[m]], row=1, col=m+1, secondary_y=True)
            fig.update_yaxes(title_text="P_m", range=[0, 1.1], row=2, col=m+1)
            fig.update_xaxes(title_text="x", row=1, col=m+1)
            fig.update_xaxes(title_text="m", row=2, col=m+1)
        fig.frames = frames
        return fig
    # single model: c_m is a single array, convert to np.array
    c_m = np.asarray(c_m)
    amp_max = np.max(np.abs(psi_xt)) * 1.1
    prob_max = np.max(np.abs(psi_xt)**2) * 1.1
    V_max = np.max(V if V.ndim > 1 else V) * 1.1
    re_max = np.max(np.abs(np.real(psi_xt))) * 1.1
    im_max = np.max(np.abs(np.imag(psi_xt))) * 1.1
    M = min(10, c_m.shape[1])
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Probability Density", "State Populations", "Real Part", "Imaginary Part"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    frames = []
    for i in range(len(times)):
        psi = psi_xt[i]
        t = times[i]
        V_frame = V[i] if V.ndim > 1 else V
        Pm = np.abs(c_m[i][:M])**2
        frame = go.Frame(
            data=[
                go.Scatter(x=x, y=np.abs(psi)**2, name="|ψ|²"),
                go.Scatter(x=x, y=V_frame, name="Potential", line=dict(dash='dash')),
                go.Bar(x=np.arange(M), y=Pm, name="Populations"),
                go.Scatter(x=x, y=np.real(psi), name="Re ψ"),
                go.Scatter(x=x, y=np.imag(psi), name="Im ψ")
            ],
            name=f"frame{i}"
        )
        frames.append(frame)
    fig.add_trace(
        go.Scatter(x=x, y=np.abs(psi_xt[0])**2, name="|ψ|²"),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=x, y=V[0] if V.ndim > 1 else V, name="Potential", line=dict(dash='dash')),
        row=1, col=1, secondary_y=True
    )
    fig.add_trace(
        go.Bar(x=np.arange(M), y=np.abs(c_m[0][:M])**2, name="Populations"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x, y=np.real(psi_xt[0]), name="Re ψ"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=np.imag(psi_xt[0]), name="Im ψ"),
        row=2, col=2
    )
    fig.update_layout(
        height=800,
        showlegend=True,
        width=None,
        sliders=[{
            'currentvalue': {
                'prefix': 'Time: ',
                'suffix': ' s',
                'xanchor': 'right',
                'font': {'size': 28}
            },
            'pad': {'t': 50},
            'len': 0.9,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top',
            'transition': {'duration': 0},
            'steps': [
                {
                    'label': f'{t:.2f}',
                    'method': 'animate',
                    'args': [[f'frame{i}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                for i, t in enumerate(times)
            ]
        }],
        updatemenus=[
            {
                'type': 'buttons',
                'showactive': False,
                'x': 0.1,
                'y': -.15,
                'xanchor': 'right',
                'yanchor': 'top',
                'pad': {'t': 0, 'r': 10},
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000/10, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            },
            {
                'type': 'buttons',
                'direction': 'right',
                'showactive': True,
                'x': 0.1,
                'y': -.3,
                'xanchor': 'left',
                'yanchor': 'top',
                'pad': {'t': 0, 'r': 10},
                'buttons': [
                    {
                        'label': f'{speed}x',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100/(speed), 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                    for speed in [1, 5, 10, 20, 50]
                ]
            }
        ]
    )
    fig.frames = frames
    # Set y-axis ranges for single model
    fig.update_yaxes(title_text="|ψ|²", range=[0, prob_max], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="V", range=[0, V_max], row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="P_m", range=[0, 1.1], row=1, col=2)
    fig.update_yaxes(title_text="Re ψ", range=[-re_max, re_max], row=2, col=1)
    fig.update_yaxes(title_text="Im ψ", range=[-im_max, im_max], row=2, col=2)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=2)
    fig.update_xaxes(title_text="m", row=1, col=2)
    return fig

def main():
    st.title("Quantum Wavefunction Evolution Simulator")
    
    # Initialize session state for animation
    if 'animation_data' not in st.session_state:
        st.session_state.animation_data = None
    
    # Sidebar for input parameters
    st.sidebar.header("Simulation Parameters")
    
    # Model type selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["sudden", "dyson", "adiabatic", "all"]
    )
    
    # Initial state selection
    n = st.sidebar.number_input("Initial State (n)", min_value=0, max_value=10, value=0)
    
    # Time parameters
    time_length = st.sidebar.number_input("Simulation Time Length (s)", min_value=0.1, max_value=30.0, value=4.0)
    N_t = st.sidebar.number_input("Number of Time Steps", min_value=100, max_value=5000, value=500)
    
    # Potential parameters
    st.sidebar.subheader("Initial Potential")
    v0_type = st.sidebar.selectbox("Initial Potential Type", ["Square Well", "Harmonic"])
    
    if v0_type == "Square Well":
        width0 = st.sidebar.number_input("Initial Well Width", min_value=1e-9, max_value=10.0, value=5.0)
        depth0 = st.sidebar.number_input("Initial Well Depth", min_value=1e-9, max_value=1e6, value=100.0)
        v0_param = ("Square Well", (width0, depth0))
    else:
        freq0 = st.sidebar.number_input("Initial Frequency", min_value=0.0, max_value=100.0, value=3.0)
        v0_param = ("Harmonic", freq0)
    
    st.sidebar.subheader("Final Potential")
    v1_type = st.sidebar.selectbox("Final Potential Type", ["Square Well", "Harmonic"])
    
    if v1_type == "Square Well":
        width1 = st.sidebar.number_input("Final Well Width", min_value=1e-9, max_value=10.0, value=5.0)
        depth1 = st.sidebar.number_input("Final Well Depth", min_value=1e-9, max_value=1e6, value=100.0)
        v1_param = ("Square Well", (width1, depth1))
    else:
        freq1 = st.sidebar.number_input("Final Frequency", min_value=0.0, max_value=100.0, value=3.0)
        v1_param = ("Harmonic", freq1)
    
    # Time parameters for evolution
    if model_type == "sudden":
        start_time = st.sidebar.number_input("Start Time", min_value=0.0, max_value=time_length, value=0.5)
        time_param = (start_time, 0)
    else:
        start_time = st.sidebar.number_input("Start Time", min_value=0.0, max_value=time_length, value=0.5)
        stop_time = st.sidebar.number_input("Stop Time", min_value=start_time, max_value=time_length, value=1.5)
        time_param = (start_time, stop_time)
    
    # Animation speed control

    
    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        # Create progress bar and status text
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # Use the main function from thing.py
        if model_type == "all":
            model_names = ["Sudden", "Dyson", "Adiabatic"]
            psi_xt_list, c_m_list, V_list, times_list = [], [], [], []
            for mtype in ["sudden", "dyson", "adiabatic"]:
                psi_xt, c_m, V = quantum_main(
                    mtype, n, v0_param, v1_param, time_param, N_t=N_t, time_length=time_length
                )
                psi_xt_list.append(psi_xt)
                c_m_list.append(c_m)
                V_list.append(V)
                times_list.append(np.linspace(0, time_length, N_t))
            N = psi_xt_list[0].shape[1]
            x_max = 5.0
            x = np.linspace(-x_max, x_max, N)
            st.session_state.animation_data = {
                'psi_xt': psi_xt_list,
                'x': x,
                'times': times_list,
                'c_m': c_m_list,
                'V': V_list,
                'model_names': model_names
            }
        else:
            psi_xt, c_m, V = quantum_main(
                model_type, n, v0_param, v1_param, time_param, N_t=N_t, time_length=time_length
            )
            N = psi_xt.shape[1]
            x_max = 5.0
            x = np.linspace(-x_max, x_max, N)
            time_arr = np.linspace(0, time_length, N_t)
            
            # Store animation data in session state
            st.session_state.animation_data = {
                'psi_xt': psi_xt,
                'x': x,
                'times': time_arr,
                'c_m': c_m,
                'V': V
            }
        
        # Clear progress bar and status text
        progress_bar.empty()
        status_text.empty()
    
    # Display animation if data is available
    if st.session_state.animation_data is not None:
        data = st.session_state.animation_data
        if model_type == "all":
            fig = create_animation(
                data['psi_xt'],
                data['x'],
                data['times'],
                data['c_m'],
                data['V'],
                rate=10,
                only_density_and_pop=True,
                model_names=data['model_names']
            )
        else:
            fig = create_animation(
                data['psi_xt'],
                data['x'],
                data['times'],
                data['c_m'],
                data['V'],
                rate=10
            )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
