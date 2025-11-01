import streamlit as st
import pandas as pd
import csv
import random
import io

# ------------------------------------------------------------
# STREAMLIT APP HEADER
# ------------------------------------------------------------
st.title("📺 TV Program Scheduling using Genetic Algorithm")
st.write("This app optimizes TV program schedules based on ratings data using a Genetic Algorithm (GA).")

# ------------------------------------------------------------
# LOAD CSV FILE
# ------------------------------------------------------------
st.subheader("1️⃣ Upload or Load CSV Dataset")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample dataset. Upload your own CSV to replace it.")
    sample_data = """Program,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
News,7.8,8.1,8.4,8.5,7.5,6.2,6.1,6.8,7.0,7.3,7.5,7.7,8.0,8.4,8.7,8.9,9.0,8.8
Drama,6.0,6.2,6.4,6.8,7.0,7.5,8.0,8.5,8.9,9.1,9.3,9.0,8.7,8.2,7.9,7.5,7.0,6.8
Sports,5.0,5.2,5.4,5.7,6.0,6.4,7.0,8.0,8.5,9.0,9.4,9.7,9.8,9.5,9.2,8.8,8.4,8.0
Movie,4.0,4.5,5.0,6.0,7.0,8.0,9.0,9.3,9.5,9.7,9.6,9.4,9.0,8.6,8.3,8.0,7.6,7.2
"""
    df = pd.read_csv(io.StringIO(sample_data))

st.dataframe(df)

# ------------------------------------------------------------
# READ DATA INTO DICTIONARY
# ------------------------------------------------------------
program_ratings = {}
for i, row in df.iterrows():
    program = row.iloc[0]
    ratings = [float(x) for x in row.iloc[1:].values]
    program_ratings[program] = ratings

all_programs = list(program_ratings.keys())
all_time_slots = list(range(6, 24))  # 6 AM to 11 PM

# ------------------------------------------------------------
# SIDEBAR: GA PARAMETERS
# ------------------------------------------------------------
st.sidebar.header("⚙️ Genetic Algorithm Settings")
GEN = st.sidebar.slider("Generations", 10, 500, 100)
POP = st.sidebar.slider("Population Size", 10, 200, 50)
CO_R = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUT_R = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2)
EL_S = st.sidebar.slider("Elitism Size", 1, 10, 2)

# ------------------------------------------------------------
# DEFINE FUNCTIONS
# ------------------------------------------------------------
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += program_ratings[program][time_slot]
    return total_rating

def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population

    return max(population, key=fitness_function)

# ------------------------------------------------------------
# RUN GENETIC ALGORITHM
# ------------------------------------------------------------
if st.button("🚀 Run Optim
