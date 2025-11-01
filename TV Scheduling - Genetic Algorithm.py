import streamlit as st
import pandas as pd
import csv
import random
import io
import requests

# ------------------------------------------------------------
# STREAMLIT APP HEADER
# ------------------------------------------------------------
st.title("📺 TV Program Scheduling using Genetic Algorithm")
st.write("This app optimizes TV program schedules based on ratings data using a Genetic Algorithm (GA).")

# ------------------------------------------------------------
# LOAD CSV FROM GITHUB
# ------------------------------------------------------------
st.subheader("1️⃣ Load Dataset")

github_url = st.text_input(
    "Enter the raw GitHub URL for your CSV file:",
    "https://raw.githubusercontent.com/username/repo/main/program_ratings.csv"
)

if st.button("Load CSV"):
    try:
        response = requests.get(github_url)
        response.raise_for_status()

        csv_content = response.text
        st.success("✅ CSV loaded successfully!")

        # Display preview
        df = pd.read_csv(io.StringIO(csv_content))
        st.dataframe(df.head())

        # Convert CSV to dictionary
        def read_csv_to_dict(csv_text):
            program_ratings = {}
            reader = csv.reader(io.StringIO(csv_text))
            header = next(reader)
            for row in reader:
                program = row[0]
                ratings = [float(x) for x in row[1:]]
                program_ratings[program] = ratings
            return program_ratings

        program_ratings_dict = read_csv_to_dict(csv_content)
        ratings = program_ratings_dict

        # ------------------------------------------------------------
        # PARAMETERS
        # ------------------------------------------------------------
        st.subheader("2️⃣ GA Parameters")

        GEN = st.slider("Number of Generations", 10, 300, 100)
        POP = st.slider("Population Size", 10, 200, 50)
        CO_R = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
        MUT_R = st.slider("Mutation Rate", 0.0, 1.0, 0.2)
        EL_S = st.slider("Elitism Size", 1, 10, 2)

        all_programs = list(ratings.keys())
        all_time_slots = list(range(6, 24))

        # ------------------------------------------------------------
        # FUNCTIONS
        # ------------------------------------------------------------
        def fitness_function(schedule):
            total_rating = 0
            for time_slot, program in enumerate(schedule):
                total_rating += ratings[program][time_slot]
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

        def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP,
                              crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):

            population = [initial_schedule]

            for _ in range(population_size - 1):
                random_schedule = initial_schedule.copy()
                random.shuffle(random_schedule)
                population.append(random_schedule)

            for _ in range(generations):
                population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
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

            population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
            return population[0]

        # ------------------------------------------------------------
        # RUN GENETIC ALGORITHM
        # ------------------------------------------------------------
        if st.button("🚀 Run Genetic Algorithm"):
            st.info("Running optimization... please wait ⏳")

            initial_schedule = all_programs.copy()
            best_schedule = genetic_algorithm(initial_schedule)
            total_rating = fitness_function(best_schedule)

            st.success("✅ Optimal Schedule Generated!")

            schedule_df = pd.DataFrame({
                "Time Slot": [f"{t:02d}:00" for t in all_time_slots[:len(best_schedule)]],
                "Program": best_schedule
            })

            st.subheader("📋 Final Optimal Schedule")
            st.dataframe(schedule_df)

            st.metric("Total Ratings", f"{total_rating:.2f}")

            # Optional: chart
            st.subheader("📊 Schedule Visualization")
            st.bar_chart(data=schedule_df.set_index("Time Slot"))

    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
else:
    st.info("Enter your GitHub CSV link and click 'Load CSV'.")
