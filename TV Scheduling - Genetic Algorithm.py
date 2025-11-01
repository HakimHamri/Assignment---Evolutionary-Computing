import streamlit as st
import pandas as pd
import csv
import random

# -----------------------------------------------
# READ CSV
# -----------------------------------------------
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings

# -----------------------------------------------
# STREAMLIT HEADER
# -----------------------------------------------
st.title("📺 TV Program Scheduling using Genetic Algorithm")
st.write("Optimize TV program schedule using a Genetic Algorithm (GA) with adjustable crossover and mutation rates.")

# -----------------------------------------------
# UPLOAD CSV
# -----------------------------------------------
uploaded_file = st.file_uploader("Upload Program Ratings CSV", type=["csv"])
if uploaded_file:
    ratings = read_csv_to_dict(uploaded_file)
    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))  # Time slots from 6AM - 11PM

    # -----------------------------------------------
    # USER INPUT PARAMETERS
    # -----------------------------------------------
    st.sidebar.header("GA Parameters")
    CO_R = st.sidebar.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8)
    MUT_R = st.sidebar.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02)
    GEN = st.sidebar.number_input("Generations", 10, 500, 100)
    POP = st.sidebar.number_input("Population Size", 10, 200, 50)
    EL_S = 2

    # -----------------------------------------------
    # DEFINE FUNCTIONS
    # -----------------------------------------------
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += ratings[program][time_slot % len(ratings[program])]
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

    def genetic_algorithm(generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
        population = [random.sample(all_programs, len(all_programs)) for _ in range(population_size)]

        for generation in range(generations):
            population.sort(key=fitness_function, reverse=True)
            new_population = population[:elitism_size]

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

        return population[0]

    # -----------------------------------------------
    # RUN BUTTON
    # -----------------------------------------------
    if st.button("Run Genetic Algorithm"):
        best_schedule = genetic_algorithm()
        total_rating = fitness_function(best_schedule)

        result_df = pd.DataFrame({
            "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
            "Program": best_schedule[:len(all_time_slots)]
        })

        st.subheader("🧾 Final Schedule")
        st.table(result_df)
        st.success(f"Total Ratings: {total_rating:.2f}")

        st.write(f"Parameters used → CO_R: {CO_R}, MUT_R: {MUT_R}")
