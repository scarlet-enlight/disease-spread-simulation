"""
Disease Spread Simulation in Urban Environment
Main simulation engine for modeling disease transmission across city districts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple
import json
from pathlib import Path

PROCESSED_DIR = Path('data/processed')
RESULTS_DIR = Path('results')
SIM_RESULTS_PATH = PROCESSED_DIR / 'simulation_results.csv'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class District:
    """Represents a city district with specific characteristics"""
    name: str
    center: Tuple[float, float]
    radius: float
    transmission_rate: float
    capacity: int
    color: str

class Person:
    """Represents an individual in the simulation"""
    def __init__(self, person_id: int, x: float, y: float):
        self.id = person_id
        self.x = x
        self.y = y
        self.status = 'susceptible'  # susceptible, infected, recovered
        self.days_infected = 0
        self.current_district = None
        self.target_district = None
        self.velocity = np.random.uniform(0.5, 1.5, 2)
        
    def move(self, districts: List[District], map_size: Tuple[float, float]):
        """Move person towards target district"""
        if self.target_district is None or np.random.random() < 0.02:
            # Choose new random district
            self.target_district = np.random.choice(districts)
        
        # Move towards target
        dx = self.target_district.center[0] - self.x
        dy = self.target_district.center[1] - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0.5:
            self.x += (dx / dist) * self.velocity[0]
            self.y += (dy / dist) * self.velocity[1]
        
        # Keep within bounds
        self.x = np.clip(self.x, 0, map_size[0])
        self.y = np.clip(self.y, 0, map_size[1])
        
    def update_status(self, recovery_days: int = 14):
        """Update infection status"""
        if self.status == 'infected':
            self.days_infected += 1
            if self.days_infected >= recovery_days:
                self.status = 'recovered'
                self.days_infected = 0

class DiseaseSimulation:
    """Main simulation class"""
    def __init__(self, n_people: int = 500, map_size: Tuple[float, float] = (100, 100)):
        self.map_size = map_size
        self.n_people = n_people
        self.day = 0
        
        # Initialize districts based on typical urban areas
        self.districts = [
            District("Office District", (30, 70), 15, 0.15, 200, '#FFE5B4'),
            District("Residential Area", (70, 70), 20, 0.05, 300, '#E0F0E0'),
            District("Shopping Center", (50, 30), 12, 0.20, 150, '#FFD0D0'),
            District("Park", (25, 25), 18, 0.02, 100, '#90EE90'),
            District("Transport Hub", (75, 30), 10, 0.25, 250, '#D0D0FF'),
        ]
        
        # Initialize people
        self.people = []
        for i in range(n_people):
            x = np.random.uniform(0, map_size[0])
            y = np.random.uniform(0, map_size[1])
            self.people.append(Person(i, x, y))
        
        # Infect initial people
        initial_infected = int(n_people * 0.02)
        for i in range(initial_infected):
            self.people[i].status = 'infected'
        
        # Statistics tracking
        self.history = {
            'day': [],
            'susceptible': [],
            'infected': [],
            'recovered': [],
            'new_infections': []
        }
        
    def get_current_district(self, person: Person) -> District:
        """Determine which district a person is currently in"""
        for district in self.districts:
            dx = person.x - district.center[0]
            dy = person.y - district.center[1]
            if np.sqrt(dx**2 + dy**2) <= district.radius:
                return district
        return None
    
    def check_infection(self, person: Person):
        """Check if person gets infected based on proximity to infected individuals"""
        if person.status != 'susceptible':
            return
        
        current_district = self.get_current_district(person)
        if current_district is None:
            return
        
        # Find nearby infected people
        for other in self.people:
            if other.status == 'infected':
                distance = np.sqrt((person.x - other.x)**2 + (person.y - other.y)**2)
                if distance < 3:  # Infection radius
                    # Probability based on district transmission rate
                    if np.random.random() < current_district.transmission_rate:
                        person.status = 'infected'
                        return
    
    def step(self):
        """Execute one simulation step (one day)"""
        self.day += 1
        new_infections = 0
        
        # Move all people
        for person in self.people:
            person.move(self.districts, self.map_size)
        
        # Check for new infections
        for person in self.people:
            old_status = person.status
            self.check_infection(person)
            if old_status == 'susceptible' and person.status == 'infected':
                new_infections += 1
        
        # Update infection status
        for person in self.people:
            person.update_status()
        
        # Record statistics
        stats = self.get_statistics()
        self.history['day'].append(self.day)
        self.history['susceptible'].append(stats['susceptible'])
        self.history['infected'].append(stats['infected'])
        self.history['recovered'].append(stats['recovered'])
        self.history['new_infections'].append(new_infections)
    
    def get_statistics(self):
        """Get current simulation statistics"""
        susceptible = sum(1 for p in self.people if p.status == 'susceptible')
        infected = sum(1 for p in self.people if p.status == 'infected')
        recovered = sum(1 for p in self.people if p.status == 'recovered')
        
        return {
            'susceptible': susceptible,
            'infected': infected,
            'recovered': recovered,
            'total': self.n_people
        }
    
    def save_results(self, filename: str = str(SIM_RESULTS_PATH)):
        """Save simulation history to CSV"""
        df = pd.DataFrame(self.history)
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    def visualize_map(self, ax=None, show_districts=True):
        """Visualize current state of the simulation"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        ax.clear()
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        ax.set_aspect('equal')
        
        # Draw districts
        if show_districts:
            for district in self.districts:
                circle = plt.Circle(district.center, district.radius, 
                                  color=district.color, alpha=0.3, label=district.name)
                ax.add_patch(circle)
                ax.text(district.center[0], district.center[1], district.name,
                       ha='center', va='center', fontsize=8, weight='bold')
        
        # Draw people
        colors = {
            'susceptible': 'green',
            'infected': 'red',
            'recovered': 'blue'
        }
        
        for status, color in colors.items():
            people_status = [p for p in self.people if p.status == status]
            if people_status:
                x = [p.x for p in people_status]
                y = [p.y for p in people_status]
                ax.scatter(x, y, c=color, s=20, alpha=0.6, label=status.capitalize())
        
        ax.legend(loc='upper right')
        ax.set_title(f'Disease Spread Simulation - Day {self.day}', fontsize=14, weight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        
        return ax

def run_simulation(days: int = 100, n_people: int = 500):
    """Run complete simulation"""
    sim = DiseaseSimulation(n_people=n_people)
    
    print(f"Starting simulation with {n_people} people for {days} days")
    print(f"Initial infected: {sum(1 for p in sim.people if p.status == 'infected')}")
    
    for day in range(days):
        sim.step()
        if day % 10 == 0:
            stats = sim.get_statistics()
            print(f"Day {day}: S={stats['susceptible']}, I={stats['infected']}, R={stats['recovered']}")
    
    # Save results
    sim.save_results()
    
    # Final visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Map visualization
    sim.visualize_map(ax1)
    
    # Time series plot
    ax2.plot(sim.history['day'], sim.history['susceptible'], 'g-', label='Susceptible', linewidth=2)
    ax2.plot(sim.history['day'], sim.history['infected'], 'r-', label='Infected', linewidth=2)
    ax2.plot(sim.history['day'], sim.history['recovered'], 'b-', label='Recovered', linewidth=2)
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Number of People')
    ax2.set_title('Disease Spread Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    final_image_path = RESULTS_DIR / 'simulation_final_state.png'
    final_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(final_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return sim

if __name__ == "__main__":
    simulation = run_simulation(days=100, n_people=500)
    print("\nSimulation complete!")
    print(f"Final statistics: {simulation.get_statistics()}")
    