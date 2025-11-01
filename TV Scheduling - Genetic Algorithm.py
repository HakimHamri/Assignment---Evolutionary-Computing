import streamlit as st
import pandas as pd
import random
import csv
import io

# ------------------------------------------------------------
# FUNCTION DEFINITIONS
# ------------------------------------------------------------

# Function to read CSV and convert to dictionary
def read_csv_to_dict(file_obj, expected_slots=None):
    program_ratings = {}
    reader = csv.reader(io.StringIO(file_obj.getvalue().decode("utf-8")))
    header = next(reader)
    
    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]
        program_ratings[program] = ratings
    
    if expected_slots and len(header) - 1 != expected_slots:
        raise ValueError(f"Expected {expected_slots} slots, but found {len(header)-1}.")
    
    return program_ratings, header[1:]

# Fitness function
def calculate_fitness(schedule, program_ratings):
    total = 0
    for slot, program in enumerate(schedule):
        total += program_ratings[program][slot]
    return total

# Selection
def selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitness_scores):
        current += fitness
        if current > pick:
            return individual

# Crossover
def crossover(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1[:]
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + [p for p in parent2 if p not in parent1[:point]]
    return child

# Mutation
def mutate(schedule, mutation_rate):
    schedule = schedule[:]
    for i in range(len(schedule)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(schedule) - 1)
            schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule

# GA main function
def genetic_algorithm(program_ratings, population_size, generations, crossover_rate, mutation_rate):
    programs = list(program_ratings.keys())
    num_slots = len(next(iter(program_ratings.values())))

    population = [random.sample(programs, num_slots) for _ in range(population_size)]
    
    best_fitness = 0
    best_schedule = None

    for _ in range(generations):
        fitness_scores = [calculate_fitness(ind, program_ratings) for ind in population]
        new_population = []

        for _ in range(population_size):
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            child = crossover(parent1, parent2, crossover_rate)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        gen_best = max(fitness_scores)
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_schedule = population[fitness_scores.index(gen_best)]

    return best_schedule, best_fitness

# ------------------------------------------------------------
# STREAMLIT INTERFACE
# ------------------------------------------------------------

st.title("📺 TV Program Scheduling using Genetic Algorithm")
st.write("This app optimizes TV program schedules based on ratings data using a Genetic Algorithm (GA).")

st.sidebar.header("⚙️ GA Parameters")
POP_SIZE = st.sidebar.slider("Population Size", 5, 100, 20)
GENS = st.sidebar.slider("Generations", 10, 500, 100)
CO_R = st.sidebar.slider("Crossover Rate", 0.0, 0.95, 0.8)
MUT_R = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.2)

st.subheader("1️⃣ Load Dataset")

use_sample = st.toggle("Use Sample Dataset", value=True)
uploaded_file = None

if not use_sample:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

T = 6  # Number of time slots
all_time_slots = ["Slot 1", "Slot 2", "Slot 3", "Slot 4", "Slot 5", "Slot 6"]

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
if use_sample:
    st.info("Using sample dataset for demonstration.")
    df_sample = pd.DataFrame({
        "Program": ["News", "Drama", "Sports", "Movie", "Talk Show", "Documentary"],
        "Slot 1": [8, 6, 7, 9, 5, 6],
        "Slot 2": [7, 8, 6, 9, 5, 7],
        "Slot 3": [9, 6, 8, 7, 6, 8],
        "Slot 4": [8, 7, 9, 6, 7, 8],
        "Slot 5": [7, 8, 6, 9, 8, 7],
        "Slot 6": [8, 9, 7, 8, 6, 9]
    })
    st.dataframe(df_sample, use_container_width=True)
    ratings_dict = df_sample.set_index("Program").T.to_dict("list")
else:
    if uploaded_file:
        try:
            ratings_dict, rating_cols = read_csv_to_dict(uploaded_file, expected_slots=T)
            st.success(f"Loaded {len(ratings_dict)} programs with {T} time slots.")
            
            # Display dataset
            st.subheader("📊 Uploaded Dataset Preview")
            uploaded_file.seek(0)
            df_display = pd.read_csv(uploaded_file)
            st.dataframe(df_display, use_container_width=True)

            # Dataset summary
            st.subheader("📈 Dataset Summary")
            program_col = df_display.columns[0]
            df_summary = df_display.copy()
            df_summary["Average Rating"] = df_summary.iloc[:, 1:].mean(axis=1)
            st.dataframe(df_summary[[program_col, "Average Rating"]], use_container_width=True)

            ratings_dict = df_display.set_index(program_col).T.to_dict("list")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        st.warning("Please upload a dataset or enable sample mode.")
        st.stop()

# ------------------------------------------------------------
# RUN GA
# ------------------------------------------------------------
if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Optimizing schedule..."):
        best_schedule, best_fitness = genetic_algorithm(
            ratings_dict, POP_SIZE, GENS, CO_R, MUT_R
        )

    st.success("✅ Optimization Complete!")
    st.subheader("🏆 Best Schedule Found")
    
    result_df = pd.DataFrame({
        "Time Slot": all_time_slots,
        "Program": best_schedule,
        "Rating": [ratings_dict[best_schedule[i]][i] for i in range(len(best_schedule))]
    })

    st.dataframe(result_df, use_container_width=True)
    st.metric("Total Fitness (Rating Sum)", f"{best_fitness:.2f}")
