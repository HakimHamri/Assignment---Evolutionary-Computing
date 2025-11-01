import streamlit as st
import csv
import random
import pandas as pd

st.title("📺 TV Scheduling Optimization using Genetic Algorithm")

# ---------------------------- File Upload ----------------------------
st.header("1️⃣ Upload Program Ratings CSV File")

uploaded_file = st.file_uploader("Upload your 'program_ratings.csv' file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    def read_csv_to_dict(file):
        program_ratings = {}
        reader = csv.reader(file.read().decode('utf-8').splitlines())
        header = next(reader)  # Skip header
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
        return program_ratings

    ratings = read_csv_to_dict(uploaded_file)

    st.success("✅ CSV file loaded successfully!")
    st.write("### Sample of Program Ratings")
    st.dataframe(pd.DataFrame(ratings).T)

    # ---------------------------- Parameters ----------------------------
    st.header("2️⃣ Set Genetic Algorithm Parameters")
    GEN = st.number_input("Generations (GEN)", 10, 1000, 100)
    POP = st.number_input("Population Size (POP)", 10, 200, 50)
    CO_R = st.slider("Crossover Rate (CO_R)", 0.0, 1.0, 0.8)
    MUT_R = st.slider("Mutation Rate (MUT_R)", 0.0, 1.0, 0.2)
    EL_S = st.number_input("Elitism Size (EL_S)", 1, 10, 2)

    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))  # 6 AM to 11 PM

    # ---------------------------- Functions ----------------------------
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

    def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
        population = [initial_schedule]
        for _ in range(population_size - 1):
            random_schedule = initial_schedule.copy()
            random.shuffle(random_schedule)
            population.append(random_schedule)

        for generation in range(generations):
            population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
            new_population = population[:elitism_size]  # Elitism

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

        best = max(population, key=lambda s: fitness_function(s))
        return best

    # ---------------------------- Run Button ----------------------------
    if st.button("🚀 Run Genetic Algorithm"):
        with st.spinner("Running optimization..."):
            initial_schedule = all_programs.copy()
            random.shuffle(initial_schedule)
            best_schedule = genetic_algorithm(initial_schedule)

            # Display final results
            final_ratings = fitness_function(best_schedule)
            schedule_df = pd.DataFrame({
                "Time Slot": [f"{t:02d}:00" for t in all_time_slots[:len(best_schedule)]],
                "Program": best_schedule
            })

            st.header("3️⃣ Optimal TV Schedule")
            st.dataframe(schedule_df)

            st.success(f"🎯 Total Ratings: {final_ratings:.2f}")

            st.download_button(
                label="📥 Download Schedule as CSV",
                data=schedule_df.to_csv(index=False),
                file_name="optimal_tv_schedule.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Please upload your 'program_ratings.csv' file to begin.")
