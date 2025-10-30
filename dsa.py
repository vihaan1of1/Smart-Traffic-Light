import streamlit as st
import heapq
import math
import random
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class TrafficNetwork:
    def __init__(self):
        self.graph = defaultdict(dict)
        self.traffic_lights = {}
        self.ambulances = {}
        self.simulation_time = 0
        
    def add_road(self, u, v, weight, traffic_light_id=None):
        """Add a road between two intersections"""
        self.graph[u][v] = weight
        self.graph[v][u] = weight
        
        if traffic_light_id:
            # Randomize cycle parameters for each traffic light
            cycle_time = random.randint(25, 35)  # Different cycle times
            green_time = random.uniform(0.35, 0.45)  # Different green durations
            yellow_time = random.uniform(0.05, 0.08)  # Different yellow durations
            phase_offset = random.uniform(0, cycle_time)  # Different starting phases
            
            self.traffic_lights[traffic_light_id] = {
                'position': (u, v),
                'state': 'green',
                'cycle_time': cycle_time,
                'green_ratio': green_time,
                'yellow_ratio': yellow_time,
                'phase_offset': phase_offset,
                'emergency_override': False,
                'override_end_time': 0
            }
    
    def add_ambulance(self, ambulance_id, start_position, destination, emergency_level=1):
        """Add an ambulance to the network"""
        self.ambulances[ambulance_id] = {
            'current_position': start_position,
            'destination': destination,
            'emergency_level': emergency_level,  # 1-3 (3 being most critical)
            'route': [],
            'progress': 0,
            'current_edge': None,
            'edge_progress': 0.0,
            'speed': 0.15 + (emergency_level * 0.1),  # Faster movement based on emergency level
            'completed': False
        }
    
    def dijkstra_shortest_path(self, start, end, ambulance_id=None):
        """Find shortest path using Dijkstra's algorithm with traffic considerations"""
        distances = {node: float('infinity') for node in self.graph}
        previous_nodes = {}
        distances[start] = 0
        priority_queue = [(0, start)]
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if current_node == end:
                break
                
            if current_distance > distances[current_node]:
                continue
                
            for neighbor, weight in self.graph[current_node].items():
                # Adjust weight based on traffic conditions and emergency status
                adjusted_weight = self._calculate_adjusted_weight(
                    current_node, neighbor, weight, ambulance_id
                )
                
                distance = current_distance + adjusted_weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        return self._reconstruct_path(previous_nodes, start, end)
    
    def _calculate_adjusted_weight(self, u, v, base_weight, ambulance_id):
        """Adjust road weight based on traffic conditions and emergency status"""
        adjusted_weight = base_weight
        
        # Check if there's a traffic light on this road
        for light_id, light_data in self.traffic_lights.items():
            if light_data['position'] == (u, v) or light_data['position'] == (v, u):
                if light_data['state'] == 'red':
                    adjusted_weight += 20  # Heavy penalty for red light
                elif light_data['state'] == 'yellow':
                    adjusted_weight += 10  # Moderate penalty for yellow light
                
                # Emergency override - reduce penalty for ambulances
                if ambulance_id and light_data['emergency_override']:
                    adjusted_weight -= 15
        
        # Random traffic variation
        traffic_variation = random.uniform(0.8, 1.2)
        adjusted_weight *= traffic_variation
        
        return max(adjusted_weight, 1)  # Ensure minimum weight of 1
    
    def _reconstruct_path(self, previous_nodes, start, end):
        """Reconstruct the path from start to end"""
        path = []
        current_node = end
        
        while current_node != start:
            path.append(current_node)
            current_node = previous_nodes.get(current_node)
            if current_node is None:
                return []  # No path found
        
        path.append(start)
        path.reverse()
        return path
    
    def update_traffic_lights(self):
        """Update traffic light states with different timing for each light"""
        current_time = self.simulation_time
        
        # Reset expired emergency overrides
        for light_id, light_data in self.traffic_lights.items():
            if light_data['emergency_override'] and current_time >= light_data['override_end_time']:
                light_data['emergency_override'] = False
        
        # Check for ambulances that need priority
        active_emergencies = False
        for ambulance_id, ambulance_data in self.ambulances.items():
            if not ambulance_data['completed'] and ambulance_data['emergency_level'] >= 2:
                active_emergencies = True
                route = ambulance_data['route']
                if len(route) > 1:
                    # Get upcoming traffic lights on the route (only next 2 edges)
                    for i in range(min(2, len(route) - 1)):
                        if ambulance_data['progress'] + i < len(route) - 1:
                            u = route[ambulance_data['progress'] + i]
                            v = route[ambulance_data['progress'] + i + 1]
                            for light_id, light_data in self.traffic_lights.items():
                                if (light_data['position'] == (u, v) or 
                                    light_data['position'] == (v, u)):
                                    # Set emergency override for this traffic light for limited time
                                    light_data['emergency_override'] = True
                                    light_data['override_end_time'] = current_time + 8  # 8 seconds override
                                    light_data['state'] = 'green'
        
        # Normal traffic light cycling (only for non-override lights)
        for light_id, light_data in self.traffic_lights.items():
            if not light_data['emergency_override']:
                # Calculate cycle position with phase offset
                cycle_time = light_data['cycle_time']
                phase_offset = light_data['phase_offset']
                cycle_position = ((current_time + phase_offset) % cycle_time) / cycle_time
                
                green_ratio = light_data['green_ratio']
                yellow_ratio = light_data['yellow_ratio']
                red_ratio = 1.0 - green_ratio - yellow_ratio
                
                if cycle_position < green_ratio:  # Green phase
                    light_data['state'] = 'green'
                elif cycle_position < green_ratio + yellow_ratio:  # Yellow phase
                    light_data['state'] = 'yellow'
                else:  # Red phase
                    light_data['state'] = 'red'
    
    def simulate_step(self):
        """Simulate one time step of the system"""
        # Update simulation time
        self.simulation_time += 1
        
        # Update ambulance positions
        for ambulance_id, ambulance_data in self.ambulances.items():
            if ambulance_data['completed']:
                continue
                
            if ambulance_data['route'] and len(ambulance_data['route']) > 1:
                current_progress = ambulance_data['progress']
                route = ambulance_data['route']
                
                # Check if we've reached the destination
                if current_progress >= len(route) - 1:
                    ambulance_data['current_position'] = ambulance_data['destination']
                    ambulance_data['completed'] = True
                    ambulance_data['current_edge'] = None
                    continue
                
                # Calculate which edge we're currently on
                current_edge_index = int(current_progress)
                
                if current_edge_index < len(route) - 1:
                    # We're moving along an edge
                    current_node = route[current_edge_index]
                    next_node = route[current_edge_index + 1]
                    
                    # Update edge progress
                    ambulance_data['edge_progress'] += ambulance_data['speed']
                    
                    # If we've completed this edge, move to next one
                    if ambulance_data['edge_progress'] >= 1.0:
                        ambulance_data['progress'] += 1
                        ambulance_data['edge_progress'] = 0.0
                        ambulance_data['current_position'] = next_node
                        
                        # Check if we reached destination
                        if ambulance_data['progress'] >= len(route) - 1:
                            ambulance_data['current_position'] = ambulance_data['destination']
                            ambulance_data['completed'] = True
                            ambulance_data['current_edge'] = None
                        else:
                            # Set up next edge
                            next_next_node = route[ambulance_data['progress'] + 1]
                            ambulance_data['current_edge'] = (next_node, next_next_node)
                    else:
                        # Still moving along the current edge
                        ambulance_data['current_edge'] = (current_node, next_node)
                        ambulance_data['current_position'] = f"Moving {current_node}→{next_node}"
        
        # Update traffic lights
        self.update_traffic_lights()

def create_sample_network():
    """Create a sample traffic network"""
    network = TrafficNetwork()
    
    # Define intersections (nodes)
    intersections = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # Add roads with weights (travel time)
    roads = [
        ('A', 'B', 5, 'TL1'), ('A', 'C', 8, 'TL2'),
        ('B', 'D', 6, 'TL3'), ('B', 'E', 7, None),
        ('C', 'F', 9, 'TL4'), ('C', 'D', 5, None),
        ('D', 'G', 4, 'TL5'), ('E', 'H', 8, 'TL6'),
        ('F', 'G', 6, None), ('G', 'H', 5, 'TL7')
    ]
    
    for u, v, weight, tl_id in roads:
        network.add_road(u, v, weight, tl_id)
    
    return network

def plot_network(network):
    """Create a visualization of the traffic network"""
    fig, ax = plt.subplots(figsize=(12, 8))
    G = nx.Graph()
    
    # Add nodes and edges
    for node in network.graph:
        G.add_node(node)
    for u in network.graph:
        for v in network.graph[u]:
            G.add_edge(u, v, weight=network.graph[u][v])
    
    # Position nodes in a grid-like layout
    pos = {
        'A': (0, 2), 'B': (2, 2), 'C': (0, 0),
        'D': (2, 0), 'E': (4, 2), 'F': (0, -2),
        'G': (2, -2), 'H': (4, 0)
    }
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge labels (weights)
    edge_labels = {(u, v): f"{d['weight']}min" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Draw traffic lights with detailed info
    for tl_id, tl_data in network.traffic_lights.items():
        u, v = tl_data['position']
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        
        # Traffic light color
        color = {'green': 'green', 'yellow': 'yellow', 'red': 'red'}[tl_data['state']]
        ax.plot(x, y, 's', markersize=15, color=color, markeredgecolor='black')
        
        # Traffic light ID and cycle info
        cycle_info = f"{tl_data['cycle_time']}s"
        ax.text(x, y + 0.2, f"{tl_id}\n{cycle_info}", ha='center', va='center', 
                fontsize=7, fontweight='bold')
        
        # Add override indicator
        if tl_data['emergency_override']:
            ax.plot(x, y - 0.2, 'o', markersize=8, color='blue', alpha=0.7)
    
    # Draw ambulances moving along edges
    for ambulance_id, ambulance_data in network.ambulances.items():
        if ambulance_data.get('completed', False):
            # Ambulance reached destination - show at destination node
            dest = ambulance_data['destination']
            if dest in pos:
                x, y = pos[dest]
                ax.plot(x, y, '^', markersize=25, color='green', markeredgecolor='darkgreen', linewidth=2)
                ax.text(x, y - 0.4, f"AMB{ambulance_id}✓", ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='green')
            continue
            
        if ambulance_data['route'] and ambulance_data.get('current_edge'):
            # Calculate position along the edge
            u, v = ambulance_data['current_edge']
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                
                # Interpolate position based on edge progress
                progress = ambulance_data['edge_progress']
                x = x1 + (x2 - x1) * progress
                y = y1 + (y2 - y1) * progress
                
                # Choose color based on emergency level
                color = 'red' if ambulance_data['emergency_level'] == 1 else 'orange' if ambulance_data['emergency_level'] == 2 else 'darkred'
                
                # Draw ambulance as a triangle moving along the line
                ax.plot(x, y, '^', markersize=20, color=color, markeredgecolor='black', linewidth=2)
                ax.text(x, y - 0.3, f"AMB{ambulance_id}", ha='center', va='center', 
                       fontsize=8, fontweight='bold', color=color)
                
                # Draw a small line showing the movement path
                ax.plot([x1, x], [y1, y], 'r--', alpha=0.5, linewidth=2)
        
        elif ambulance_data['current_position'] in pos:
            # Ambulance is at an intersection
            x, y = pos[ambulance_data['current_position']]
            color = 'red' if ambulance_data['emergency_level'] == 1 else 'orange' if ambulance_data['emergency_level'] == 2 else 'darkred'
            ax.plot(x, y, '^', markersize=20, color=color, markeredgecolor='black', linewidth=2)
            ax.text(x, y - 0.3, f"AMB{ambulance_id}", ha='center', va='center', 
                   fontsize=8, fontweight='bold', color=color)
    
    plt.title("Smart Traffic Light Control System - Asynchronous Traffic Lights")
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def main():
    st.set_page_config(page_title="Smart Traffic Light Controller", layout="wide")
    
    st.title("🚦 Smart Traffic Light Controller")
    st.subheader("Using Dijkstra's Algorithm to Prioritize Emergency Vehicles")
    
    # Initialize session state
    if 'network' not in st.session_state:
        st.session_state.network = create_sample_network()
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'simulation_speed' not in st.session_state:
        st.session_state.simulation_speed = 1.0
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # Simulation speed control
        st.subheader("Simulation Settings")
        st.session_state.simulation_speed = st.slider("Simulation Speed", 0.1, 3.0, 1.0, 0.1)
        
        # Ambulance management
        st.subheader("Ambulance Management")
        ambulance_id = st.selectbox("Ambulance ID", [1, 2, 3])
        start_pos = st.selectbox("Start Position", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        dest_pos = st.selectbox("Destination", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        emergency_level = st.slider("Emergency Level", 1, 3, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add/Update Ambulance"):
                route = st.session_state.network.dijkstra_shortest_path(start_pos, dest_pos, ambulance_id)
                if route:
                    st.session_state.network.add_ambulance(
                        ambulance_id, start_pos, dest_pos, emergency_level
                    )
                    st.session_state.network.ambulances[ambulance_id]['route'] = route
                    st.session_state.network.ambulances[ambulance_id]['progress'] = 0
                    st.session_state.network.ambulances[ambulance_id]['edge_progress'] = 0.0
                    st.session_state.network.ambulances[ambulance_id]['completed'] = False
                    if len(route) > 1:
                        st.session_state.network.ambulances[ambulance_id]['current_edge'] = (route[0], route[1])
                    st.success(f"Ambulance {ambulance_id} added with route: {' → '.join(route)}")
                else:
                    st.error(f"No valid route found from {start_pos} to {dest_pos}")
        
        with col2:
            if st.button("Remove Ambulance"):
                if ambulance_id in st.session_state.network.ambulances:
                    del st.session_state.network.ambulances[ambulance_id]
                    st.success(f"Ambulance {ambulance_id} removed")
        
        # Simulation controls
        st.subheader("Simulation Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Simulation" if not st.session_state.simulation_running else "Pause Simulation"):
                st.session_state.simulation_running = not st.session_state.simulation_running
        
        with col2:
            if st.button("Reset Simulation"):
                st.session_state.network = create_sample_network()
                st.session_state.simulation_running = False
                st.rerun()
        
        # Traffic light info
        st.subheader("Traffic Light Info")
        st.markdown("""
        - **Different cycle times**: 25-35 seconds
        - **Phase offsets**: Different start times
        - **Variable durations**: Different green/yellow/red ratios
        - **Blue dot**: Emergency override active
        - **Numbers**: Cycle time in seconds
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Traffic Network Visualization")
        fig = plot_network(st.session_state.network)
        st.pyplot(fig)
        
        # Show ambulance movement details
        st.subheader("Ambulance Movement Details")
        for amb_id, amb_data in st.session_state.network.ambulances.items():
            if amb_data.get('completed', False):
                st.success(f"**Ambulance {amb_id}**: ✅ Reached destination {amb_data['destination']}")
            elif amb_data['route']:
                current_edge = amb_data.get('current_edge', None)
                total_edges = len(amb_data['route']) - 1
                current_edge_index = amb_data['progress']
                
                # Safely handle current_edge display
                if current_edge:
                    edge_display = f"{current_edge[0]} → {current_edge[1]}"
                else:
                    edge_display = "Starting position"
                
                progress_percent = ((current_edge_index + amb_data['edge_progress']) / total_edges * 100) if total_edges > 0 else 0
                
                st.write(f"**Ambulance {amb_id}**: {edge_display} "
                        f"(Edge {current_edge_index+1}/{total_edges}) "
                        f"- Progress: {progress_percent:.1f}% "
                        f"- Speed: {amb_data['speed']:.2f}x")
            else:
                st.write(f"**Ambulance {amb_id}**: No route assigned")
    
    with col2:
        st.subheader("System Status")
        
        # Traffic light status with detailed info
        st.write("### Traffic Lights Status")
        traffic_light_data = []
        for tl_id, tl_data in st.session_state.network.traffic_lights.items():
            # Calculate time in current state
            cycle_time = tl_data['cycle_time']
            phase_offset = tl_data['phase_offset']
            current_cycle_pos = ((st.session_state.network.simulation_time + phase_offset) % cycle_time) / cycle_time
            
            traffic_light_data.append({
                'TL': tl_id,
                'State': tl_data['state'],
                'Cycle': f"{tl_data['cycle_time']}s",
                'Override': '🔵' if tl_data['emergency_override'] else '⚪',
                'Position': f"{tl_data['position'][0]}-{tl_data['position'][1]}"
            })
        
        if traffic_light_data:
            st.dataframe(pd.DataFrame(traffic_light_data))
        
        # Ambulance status
        st.write("### Ambulance Status")
        ambulance_data = []
        for amb_id, amb_data in st.session_state.network.ambulances.items():
            if amb_data.get('completed', False):
                status = "✅ Completed"
                progress = "100%"
            else:
                total_edges = len(amb_data['route']) - 1 if amb_data['route'] else 1
                progress_percent = min((amb_data['progress'] + amb_data['edge_progress']) / max(total_edges, 1) * 100, 100)
                status = "Moving" if amb_data.get('current_edge') else "Waiting"
                progress = f"{progress_percent:.1f}%"
            
            ambulance_data.append({
                'Ambulance': f"AMB{amb_id}",
                'Status': status,
                'Destination': amb_data['destination'],
                'Emergency': amb_data['emergency_level'],
                'Speed': f"{amb_data['speed']:.2f}x",
                'Progress': progress
            })
        
        if ambulance_data:
            st.dataframe(pd.DataFrame(ambulance_data))
        else:
            st.info("No ambulances active")
        
        # Real-time traffic light states
        st.write("### Current Traffic Light States")
        light_states = {'green': 0, 'yellow': 0, 'red': 0}
        for tl_data in st.session_state.network.traffic_lights.values():
            light_states[tl_data['state']] += 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 Green", light_states['green'])
        with col2:
            st.metric("🟡 Yellow", light_states['yellow'])
        with col3:
            st.metric("🔴 Red", light_states['red'])
    
    # Simulation update
    if st.session_state.simulation_running:
        st.session_state.network.simulate_step()
        time.sleep(0.5 / st.session_state.simulation_speed)  # Adjust speed
        st.rerun()
    
    # Statistics
    st.subheader("System Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Traffic Lights", len(st.session_state.network.traffic_lights))
    with col2:
        st.metric("Active Ambulances", len(st.session_state.network.ambulances))
    with col3:
        completed_ambulances = sum(1 for amb in st.session_state.network.ambulances.values() 
                                 if amb.get('completed', False))
        st.metric("Completed Trips", completed_ambulances)
    with col4:
        override_lights = sum(1 for tl in st.session_state.network.traffic_lights.values() 
                            if tl['emergency_override'])
        st.metric("Emergency Overrides", override_lights)

if __name__ == "__main__":
    main()