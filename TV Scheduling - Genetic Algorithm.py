import streamlit as st
import pandas as pd
import csv
import random
import io

# ------------------------------------------------------------
# APP TITLE AND INTRODUCTION
# ------------------------------------------------------------
st.title("📺 TV Program Scheduling using Genetic Algorithm")
st.write("""
This app optimizes TV program schedules based on ratings data using a **Genetic Algorithm (GA)**.
Upload your CSV file containing program ratings to get started.
""")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
st.subheader("1️⃣ Load Dataset")

uploaded_file = st.file_uploader("Upload your `program_ratings.csv` file", type=["csv"])

if uploaded_file:
    # Read CSV into dictionary
    def read_csv_to_dict(file):
        program_ratings = {}
        reader = csv.reader(io.StringIO(file.getvalue().decode("utf-8")))
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
        return program_ratings, header

    program_ratings_dict, header = read_csv_to_dict(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(df)

    # ------------------------------------------------------------
    # GA PARAMETERS
    # ------------------------------------------------------------
    st.subheader("2️⃣ Configure Genetic Algorithm Parameters")

    GEN = st.slider("Number of Generations", 10, 500, 100)
    POP = st.slider("Population Size", 10, 200, 50)
    CO_R = st.slider("Crossover Rate", 0.1, 1.0, 0.8)
    MUT_R = st.slider("Mutation Rate", 0.0, 1.0, 0.2)
    EL_S = st.slider("Elitism Size", 1, 10, 2)

    all_programs = list(program_ratings_dict.keys())
    all_time_slots = list(range(6, 24))  # 6 AM to 11 PM

    # ------------------------------------------------------------
    # DEFINE FUNCTIONS
    # ------------------------------------------------------------
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += program_ratings_dict[program][time_slot % len(program_ratings_dict[program])]
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
            population.sort(key=lambda s: fitness_function(s), reverse=True)
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

            population = new_population[:population_size]

        return max(population, key=lambda s: fitness_function(s))

    # ------------------------------------------------------------
    # RUN GENETIC ALGORITHM
    # ------------------------------------------------------------
    st.subheader("3️⃣ Run Optimization")

    if st.button("🚀 Run Genetic Algorithm"):
        initial_schedule = all_programs.copy()
        random.shuffle(initial_schedule)

        best_schedule = genetic_algorithm(initial_schedule)
        best_fitness = fitness_function(best_schedule)

        # Display results
        st.success("✅ Optimization Complete!")
        st.write("### 🏆 Final Optimal Schedule:")

        schedule_data = {
            "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
            "Program": [best_schedule[i % len(best_schedule)] for i in range(len(all_time_slots))],
        }
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df)

        st.metric(label="Total Ratings", value=round(best_fitness, 2))

        st.download_button(
            label="💾 Download Schedule as CSV",
            data=schedule_df.to_csv(index=False).encode("utf-8"),
            file_name="optimal_schedule.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Please upload a CSV file to start the optimization.")
